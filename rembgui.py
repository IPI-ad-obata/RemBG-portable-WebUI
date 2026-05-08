import gradio as gr
import numpy as np
import cv2
import json
import os
from rembg import remove, new_session
from PIL import Image
import webbrowser
from threading import Timer

# カレントディレクトリに「models」フォルダを作成してそこを使うように強制
models_dir = os.path.join(os.getcwd(), "models")
os.makedirs(models_dir, exist_ok=True)
os.environ["U2NET_HOME"] = models_dir

# --- 定数・モデル設定 ---
MODELS = ["birefnet-massive", "birefnet-general-lite", "birefnet-portrait", "birefnet-dis", "birefnet-hrsod", "isnet-general-use", "isnet-anime", "sam"]

def on_model_change(model_name):
    is_sam = model_name == "sam"
    return gr.update(interactive=is_sam), gr.update(interactive=is_sam), gr.update(interactive=is_sam)

def update_alpha_ui(is_enabled):
    return gr.update(interactive=is_enabled), gr.update(interactive=is_enabled), gr.update(interactive=is_enabled)

# --- 画像処理ロジック ---
def apply_background_logic(image_np, bg_mode):
    if image_np is None: return None
    pil_img = Image.fromarray(image_np)
    if bg_mode == "🏁透明(PNG/WebP)": return image_np
    
    bg_color = (255, 255, 255) if bg_mode == "⚪白背景" else (0, 0, 0)
    canvas = Image.new("RGB", pil_img.size, bg_color)
    if pil_img.mode == 'RGBA':
        canvas.paste(pil_img, (0, 0), pil_img)
    else:
        canvas.paste(pil_img, (0, 0))
    return np.array(canvas)

def prepare_file(result_rgba, format_type, bg_mode):
    if result_rgba is None: return None
    pil_img = Image.fromarray(result_rgba)
    file_path = f"output.{format_type.lower()}"
    
    # 見たまま保存のロジック
    need_composite = bg_mode != "🏁透明(PNG/WebP)" or format_type == "JPEG"
    target_bg = (0, 0, 0) if bg_mode == "⚫黒背景" else (255, 255, 255)

    if need_composite:
        final_img = Image.new("RGB", pil_img.size, target_bg)
        if pil_img.mode == 'RGBA':
            final_img.paste(pil_img, mask=pil_img.split()[3])
        else:
            final_img.paste(pil_img)
    else:
        final_img = pil_img

    final_img.save(file_path, format_type, quality=95)
    return file_path

def run_rembg(clean_img, model_name, sam_json, alpha_matting, ero, fgt, bgt, bg_mode, format_type):
    if clean_img is None: return None, None, None
    if model_name == "sam":
        session = new_session(model_name, sam_model="sam_vit_h_4b8939", sam_quant=True)
    else:
        session = new_session(model_name)
    
    kwargs = {
        "alpha_matting": alpha_matting,
        "alpha_matting_foreground_threshold": fgt,
        "alpha_matting_background_threshold": bgt,
        "alpha_matting_erode_size": ero
    }
    if model_name == "sam" and sam_json:
        try: kwargs["extra_params"] = json.loads(sam_json)
        except: pass

    result_rgba = remove(clean_img, session=session, **kwargs)
    preview = apply_background_logic(result_rgba, bg_mode)
    file = prepare_file(result_rgba, format_type, bg_mode)
    return result_rgba, preview, file

# --- インタラクション ---
def get_coords(evt: gr.SelectData, current_points, mode, clean_img):
    if clean_img is None: return None, current_points, ""
    x, y = evt.index
    current_points.append([x, y, 1 if mode == "前景 (赤)" else 0])
    
    annotated_img = clean_img.copy()
    for px, py, pl in current_points:
        color = (255, 0, 0) if pl == 1 else (0, 0, 255) # 前景赤、背景青
        cv2.circle(annotated_img, (px, py), 7, color, -1)
        cv2.circle(annotated_img, (px, py), 8, (255, 255, 255), 1)
        
    res_json = json.dumps({"prompt_points": [[p[0], p[1]] for p in current_points], "prompt_labels": [p[2] for p in current_points]})
    return annotated_img, current_points, res_json

def open_browser():
    # Gradioのデフォルトポートは7860。もし他で使っているなら変更が必要
    webbrowser.open_new("http://127.0.0.1:7860")

def check_image_upload(img):
    if img is None:
        return None, [], ""
    
    # ここでコンソール（黒い画面）にサイズを表示
    print(f"--- 画像アップロード検知 ---")
    print(f"解像度: {img.shape[1]} x {img.shape[0]} (横 x 縦)")
    print(f"チャンネル数: {img.shape[2]}")
    print(f"---------------------------")
    
    return img, [], ""

# --- UI Layout ---
with gr.Blocks(title="Production RemBG") as demo:
    gr.Markdown("## AI切り抜きツール")
    pts_state = gr.State([])
    clean_img_state = gr.State()
    result_state = gr.State()
    
    with gr.Row():
        with gr.Column():
            input_view = gr.Image(label="入力プレビュー", type="numpy", sources=["upload"])
            model_sel = gr.Dropdown(MODELS, value="isnet-general-use", label="AIモデル")
            
            with gr.Accordion("詳細設定 (Alpha Matting)", open=False):
                alpha_sw = gr.Checkbox(label="有効化", value=False)
                with gr.Row():
                    erode_size = gr.Slider(0, 40, value=10, step=1, label="Erode Size", interactive=False)
                    fg_thresh = gr.Slider(0, 255, value=240, step=1, label="FG Threshold", interactive=False)
                    bg_thresh = gr.Slider(0, 255, value=10, step=1, label="BG Threshold", interactive=False)
            
            with gr.Group():
                mode = gr.Radio(["前景 (赤)", "背景 (青)"], value="前景 (赤)", label="SAMモード", interactive=False)
                json_info = gr.Textbox(label="SAM座標データ", interactive=False)
                clear_btn = gr.Button("ポイントリセット", interactive=False)
            
            run_btn = gr.Button("背景削除を実行", variant="primary")

        with gr.Column():
            result_view = gr.Image(label="出力プレビュー", buttons=["fullscreen"])
            bg_selector = gr.Radio(["🏁透明(PNG/WebP)", "⚪白背景", "⚫黒背景"], value="🏁透明(PNG/WebP)", label="プレビュー背景色")
            with gr.Row():
                format_sel = gr.Dropdown(["PNG", "JPEG", "WebP"], value="PNG", label="保存形式")
                dl_btn = gr.DownloadButton("ダウンロード", variant="primary")

    # --- Events ---
    input_view.upload(lambda img: (img, [], ""), inputs=[input_view], outputs=[clean_img_state, pts_state, json_info])
    model_sel.change(on_model_change, model_sel, [mode, json_info, clear_btn])
    alpha_sw.change(update_alpha_ui, [alpha_sw], [erode_size, fg_thresh, bg_thresh])
    
    input_view.select(get_coords, [pts_state, mode, clean_img_state], [input_view, pts_state, json_info])
    clear_btn.click(lambda img: (img, [], ""), inputs=[clean_img_state], outputs=[input_view, pts_state, json_info])
    
    run_btn.click(run_rembg, [clean_img_state, model_sel, json_info, alpha_sw, erode_size, fg_thresh, bg_thresh, bg_selector, format_sel], [result_state, result_view, dl_btn])

    # リアルタイム更新
    bg_selector.change(apply_background_logic, [result_state, bg_selector], result_view)
    bg_selector.change(prepare_file, [result_state, format_sel, bg_selector], dl_btn)
    format_sel.change(prepare_file, [result_state, format_sel, bg_selector], dl_btn)

if __name__ == "__main__":
    # 1.2秒後にブラウザを開く予約（サーバー起動を待つため）
    Timer(1.2, open_browser).start()
    
    # サーバー起動
    demo.launch()