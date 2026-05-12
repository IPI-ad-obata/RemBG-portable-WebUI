import gradio as gr
import numpy as np
import cv2
import json
import os
from rembg import remove, new_session
from PIL import Image, ImageDraw
import webbrowser
from threading import Timer

# カレントディレクトリに「models」フォルダを作成してそこを使うように強制
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
os.environ["U2NET_HOME"] = models_dir

# --- 定数・モデル設定 ---
MODELS = ["birefnet-massive", "birefnet-general-lite", "birefnet-portrait", "birefnet-dis", "birefnet-hrsod", "isnet-general-use", "isnet-anime", "sam"]

# --- 1. Python: 座標管理とUI制御 ---

def on_model_change(model_name):
    """モデル変更時にモードを更新（SAM以外は強制的に範囲選択へ）"""
    if model_name == "sam":
        return gr.update(choices=["範囲選択(2点クリック)", "前景(赤)", "背景(青)"])
    else:
        return gr.update(choices=["範囲選択(2点クリック)"], value="範囲選択(2点クリック)")

def get_coords(evt: gr.SelectData, current_points, box_coords, mode, clean_img, model_name):
    if clean_img is None:
        return gr.update(), current_points, box_coords, ""

    x, y = evt.index
    annotated_img = clean_img.copy()
    draw = ImageDraw.Draw(annotated_img)

    # --- 共通の範囲指定処理 ---
    if mode == "範囲選択(2点クリック)":
        if len(box_coords) < 2:
            box_coords.append([x, y])
        else:
            box_coords = [[x, y]] # 3回目でリセット

    # --- SAM専用の点指定処理 ---
    elif model_name == "sam" and mode in ["前景(赤)", "背景(青)"]:
        current_points.append([x, y, 1 if mode == "前景(赤)" else 0])

    # --- 描画レイヤーの構築 (全モード共通) ---
    # 1. 範囲枠の描画
    if len(box_coords) > 0:
        for i, pt in enumerate(box_coords):
            draw.ellipse([pt[0]-5, pt[1]-5, pt[0]+5, pt[1]+5], fill=(0, 255, 160), outline=(255, 255, 255))
        if len(box_coords) == 2:
            x1, y1 = box_coords[0]
            x2, y2 = box_coords[1]
            draw.rectangle([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], outline=(0, 255, 160), width=3)

    # 2. 点の描画 (SAMのみ)
    if model_name == "sam":
        for px, py, pl in current_points:
            color = (255, 0, 0) if pl == 1 else (0, 0, 255)
            draw.ellipse([px-7, py-7, px+7, py+7], fill=color, outline=(255, 255, 255))

    # JSONデータの更新 (SAM用)
    res_json = ""
    if model_name == "sam" and (len(current_points) > 0 or len(box_coords) == 2):
        data = {}
        if len(current_points) > 0:
            data["prompt_points"] = [[p[0], p[1]] for p in current_points]
            data["prompt_labels"] = [p[2] for p in current_points]
        # SAMにはboxとして座標を渡す
        if len(box_coords) == 2:
            x1, y1 = box_coords[0]
            x2, y2 = box_coords[1]
            data["box"] = [min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)]
        res_json = json.dumps(data)

    return annotated_img, current_points, box_coords, res_json

# --- 2. 実行ロジック ---
def run_rembg(clean_img, model_name, box_coords, sam_pts_json, alpha_sw, ero, fgt, bgt, bg_mode, format_type):
    if clean_img is None: return None, None, None
    
    # 1. 範囲座標 [x1, y1, x2, y2] の確定
    box = None
    crop_offset_x, crop_offset_y = 0, 0
    if len(box_coords) == 2:
        x1, y1 = box_coords[0]
        x2, y2 = box_coords[1]
        box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        crop_offset_x, crop_offset_y = int(box[0]), int(box[1])

    session = new_session(model_name, sam_model="sam_vit_h_4b8939", sam_quant=True) if model_name == "sam" else new_session(model_name)
    kwargs = {"alpha_matting": alpha_sw, "alpha_matting_foreground_threshold": fgt, "alpha_matting_background_threshold": bgt, "alpha_matting_erode_size": ero}

    # 2. 推論用画像の準備（範囲指定があればクロップ）
    working_img = clean_img.crop(box) if box else clean_img

    if model_name == "sam":
        # 3. SAM用座標プロンプトの補正（画像全体座標 -> クロップ内座標）
        if sam_pts_json:
            try:
                params = json.loads(sam_pts_json)
                # 前景/背景点の座標を補正: [x - x1, y - y1]
                if "prompt_points" in params:
                    params["prompt_points"] = [
                        [p[0] - crop_offset_x, p[1] - crop_offset_y] for p in params["prompt_points"]
                    ]
                # プロンプト用のBox座標も補正: [x1-x1, y1-y1, x2-x1, y2-y1]
                if "box" in params:
                    b = params["box"]
                    params["box"] = [b[0] - crop_offset_x, b[1] - crop_offset_y, b[2] - crop_offset_x, b[3] - crop_offset_y]
                
                kwargs["extra_params"] = params
            except Exception as e:
                print(f"JSON Parse Error: {e}")

    # 4. 推論実行
    result_rgba = remove(working_img, session=session, **kwargs)

    # 5. クロップしていた場合、元の画像サイズに貼り戻す
    if box:
        final_rgba = Image.new("RGBA", clean_img.size, (0, 0, 0, 0))
        final_rgba.paste(result_rgba, (crop_offset_x, crop_offset_y))
        result_rgba = final_rgba

    preview = apply_background_logic(result_rgba, bg_mode)
    file = prepare_file(result_rgba, format_type, bg_mode)
    return np.array(result_rgba), preview, file

# (apply_background_logic, prepare_file は前回同様)
def apply_background_logic(image_rgba, bg_mode):
    if image_rgba is None: return None
    pil_img = Image.fromarray(image_rgba) if isinstance(image_rgba, np.ndarray) else image_rgba
    if bg_mode == "🏁透明(PNG/WebP)": return np.array(pil_img)
    bg_color = (255, 255, 255) if bg_mode == "白背景" else (0, 0, 0)
    canvas = Image.new("RGB", pil_img.size, bg_color)
    if pil_img.mode == 'RGBA': canvas.paste(pil_img, (0, 0), pil_img)
    else: canvas.paste(pil_img, (0, 0))
    return np.array(canvas)

def prepare_file(result_rgba, format_type, bg_mode):
    if result_rgba is None: return None
    pil_img = Image.fromarray(result_rgba) if isinstance(result_rgba, np.ndarray) else result_rgba
    file_path = f"output.{format_type.lower()}"
    need_composite = bg_mode != "🏁透明(PNG/WebP)" or format_type == "JPEG"
    target_bg = (0, 0, 0) if bg_mode == "黒背景" else (255, 255, 255)
    if need_composite:
        final_img = Image.new("RGB", pil_img.size, target_bg)
        final_img.paste(pil_img, mask=pil_img.split()[3]) if pil_img.mode == 'RGBA' else final_img.paste(pil_img)
    else: final_img = pil_img
    final_img.save(file_path, format_type, quality=95)
    return file_path

# --- 3. UI ---
with gr.Blocks() as demo:
    clean_img_state = gr.State()
    pts_state = gr.State([])
    box_state = gr.State([])
    result_state = gr.State()

    with gr.Row():
        with gr.Column():
            input_i = gr.Image(type="pil", label="入力プレビュー", interactive=True, sources=["upload"])
            
            with gr.Group():
                model_sel = gr.Dropdown(MODELS, value="birefnet-general-lite", label="モデル選択")
                mode_radio = gr.Radio(["範囲選択(2点クリック)"], value="範囲選択(2点クリック)", label="操作モード(2点クリック)")

            with gr.Accordion("Alpha Matting", open=False):
                alpha_sw = gr.Checkbox(label="Alpha Matting", value=False)
                with gr.Row():
                        erode_size = gr.Slider(0, 30, value=10, step=1, label="Erode Size", interactive=False)
                        fg_thresh = gr.Slider(0, 255, value=240, step=1, label="FG Threshold", interactive=False)
                        bg_thresh = gr.Slider(0, 255, value=10, step=1, label="BG Threshold", interactive=False)
           
            with gr.Row():
                reset_btn = gr.Button("リセット / 選択範囲初期化", variant="secondary")
                run_btn = gr.Button("背景削除を実行", variant="primary")
                sam_json = gr.Textbox(label="SAM プロンプトJSON (デバッグ用)", visible=False)
                
        with gr.Column():
            output_view = gr.Image(type="numpy", label="出力プレビュー", buttons=["fullscreen"])
            bg_selector = gr.Radio(["🏁透明(PNG/WebP)", "白背景", "黒背景"], value="🏁透明(PNG/WebP)", label="プレビュー背景")
            format_sel = gr.Radio(["PNG", "WebP", "JPEG"], value="PNG", label="保存形式")
            dl_btn = gr.DownloadButton("ダウンロード", variant="primary")

    # --- Events ---
    model_sel.change(on_model_change, inputs=[model_sel], outputs=[mode_radio])
    input_i.upload(lambda img: (img, img, [], [], ""), inputs=[input_i], outputs=[clean_img_state, input_i, pts_state, box_state, sam_json])
    input_i.select(get_coords, [pts_state, box_state, mode_radio, clean_img_state, model_sel], [input_i, pts_state, box_state, sam_json])
    reset_btn.click(lambda img: (img, img, [], [], ""), inputs=[clean_img_state], outputs=[clean_img_state, input_i, pts_state, box_state, sam_json])
    run_btn.click(run_rembg, [clean_img_state, model_sel, box_state, sam_json, alpha_sw, erode_size, fg_thresh, bg_thresh, bg_selector, format_sel], [result_state, output_view, dl_btn])

    # リアルタイム更新
    bg_selector.change(apply_background_logic, [result_state, bg_selector], output_view)
    bg_selector.change(prepare_file, [result_state, format_sel, bg_selector], dl_btn)
    format_sel.change(prepare_file, [result_state, format_sel, bg_selector], dl_btn)

if __name__ == "__main__":
    Timer(1.5, lambda: webbrowser.open_new("http://127.0.0.1:7860")).start()
    demo.launch()
