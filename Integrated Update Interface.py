import gradio as gr
import torch
import time
import numpy as np


# --- 導入我們之前寫好的核心模組 (此處使用 Mock class 模擬串聯) ---
# 實際執行時，可直接 import 你之前檔案中的 CREDTS_v2_Model, CREDTSScraperPipeline
class MockCREDTSInterface:
    def __init__(self):
        self.current_loss = 0.8542
        self.trained_epochs = 12

    def run_pipeline(self, url):
        """ 模擬爬取、NLP tokenization 到預測的完整流 """
        time.sleep(1.5)  # 模擬網頁爬取與 Transformer 運算延遲

        # 模擬 NLP 抓取到的暗物質關鍵特徵
        detected_signals = ["foreclosure suit filed", "maturity wall breach",
                            "SOFR rate shock 7.15%"] if "legal" in url or "http" in url else ["stable lease structure"]

        # 模擬 DeepSurv 輸出的 log-hazard ratio
        risk_score = 2.84 if "legal" in url or "http" in url else -0.45

        # 模擬產生生存曲線數據 S(t) 軸
        months = np.arange(1, 61)
        # 如果風險高，生存機率隨時間劇烈非線性下滑
        if risk_score > 0:
            survival_prob = np.exp(-0.02 * months * np.exp(0.5 * risk_score))
        else:
            survival_prob = np.exp(-0.005 * months)

        plot_data = [[m, p] for m, p in zip(months, survival_prob)]

        return risk_score, ", ".join(detected_signals), plot_data

    def fine_tune_model(self):
        """ 模擬讀取爬取的替代數據並即時更新 DeepSurv 權重 """
        time.sleep(2.0)  # 模擬 PyTorch 反向傳播訓練
        self.trained_epochs += 1
        self.current_loss -= 0.0412  # 模擬損失函數下降
        return f"Fine-tuning complete. Epochs: {self.trained_epochs} | Current DeepSurv Loss: {self.current_loss:.4f}"


# 初始化核心架構
engine = MockCREDTSInterface()

# --- 構建 Gradio 科技感介面 ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🏢 **CRE-DTS v2.0: Institutional Risk Surveillance & Model Update Dashboard**")
    gr.Markdown(
        "### *Integrating Alternative Data Scraping, Transformer NLP, and DeepSurv Neural Networks for Systemic Financial Protection*")

    with gr.Tab("🎯 Dynamic Risk Inference (即時風險預警系統)"):
        gr.Markdown("#### Enter a public legal registry or municipal docket URL to extract hidden distress vectors.")

        with gr.Row():
            url_input = gr.Textbox(
                label="Target Municipal/Legal Notice URL",
                placeholder="https://www.cookcountyclerkil.gov/legal-notices/docket-2026-993",
                scale=4
            )
            analyze_btn = gr.Button("Execute CRE-DTS Pipeline", variant="primary", scale=1)

        with gr.Row():
            with gr.Column():
                risk_output = gr.Number(label="Predicted Log-Hazard Ratio (h_theta(X))")
                signals_output = gr.Textbox(label="Extracted NLP Alternative Tokens")
            with gr.Column():
                # 繪製動態生存曲線圖表
                curve_plot = gr.LinePlot(
                    x="Month", y="Survival Probability S(t)",
                    title="Asset Survival Curve Projection (Maturity Horizon)",
                    tooltip=["Month", "Survival Probability S(t)"],
                    width=450, height=250
                )

        analyze_btn.click(
            fn=engine.run_pipeline,
            inputs=url_input,
            outputs=[risk_output, signals_output, curve_plot]
        )

    with gr.Tab("🔄 Algorithmic Self-Correction & Updates (模型在線更新與微調)"):
        gr.Markdown(
            "#### Feed newly scraped legal text batches into the Transformer layer and update the DeepSurv weights via continuous optimization loops.")

        with gr.Row():
            with gr.Column():
                gr.Markdown("##### **Model Status Matrix**")
                status_box = gr.Textbox(
                    value=f"Base Engine: Karpathy-style Transformer + DeepSurv\nInitial Epochs: {engine.trained_epochs}\nCurrent Tracking Loss: {engine.current_loss}",
                    label="Current Model Hyperparameters",
                    lines=4
                )
            with gr.Column():
                gr.Markdown("##### **Trigger Gradient Descent**")
                update_btn = gr.Button("Execute Online Fine-Tuning (AdamW Backprop)", variant="stop")
                update_status = gr.Markdown("*Awaiting pipeline signal activation...*")

        update_btn.click(
            fn=engine.fine_tune_model,
            inputs=None,
            outputs=status_box
        )

# 執行本地端介面
if __name__ == "__main__":
    # share=True 會生成一個限時 72 小時的公開網址，可以發給教授或推薦人看！
    demo.launch(share=False)