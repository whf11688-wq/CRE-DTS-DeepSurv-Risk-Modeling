import requests
from bs4 import BeautifulSoup
import re
import torch


class CREDTSScraperPipeline:
    """
    Automated Data Scraping & Tokenization Pipeline for CRE-DTS v2.0.
    Simulates institutional data ingestion from public legal and default registries.
    """

    def __init__(self, vocab=None):
        # Build a miniature domain-specific vocabulary map for demo purposes
        # In production, this would be replaced by tiktoken or a trained BPE tokenizer
        if vocab is None:
            self.vocab = {
                "<PAD>": 0, "<UNK>": 1, "default": 2, "foreclosure": 3,
                "commercial": 4, "mortgage": 5, "breach": 6, "suit": 7,
                "bank": 8, "interest": 9, "maturity": 10, "receiver": 11
            }
        else:
            self.vocab = vocab

        self.inv_vocab = {v: k for k, v in self.vocab.items()}

    def scrape_legal_notice(self, target_url):
        """
        Scrapes unstructured legal distress text from a target municipal registry public feed.
        """
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        }

        print(f"[LOG] Initiating HTTP Request to Target Registry: {target_url}")
        try:
            # In a live setup, this points to public dockets or legal notice aggregators (e.g., state public notice feeds)
            response = requests.get(target_url, headers=headers, timeout=10)
            if response.status_code != 200:
                print(f"[ERROR] Connection failed with Status Code: {response.status_code}")
                return None

            soup = BeautifulSoup(response.text, 'html.parser')

            # Extract text content from common article or docket summary structures
            text_blocks = []
            for paragraph in soup.find_all(['p', 'div'],
                                           class_=re.compile(re.escape('notice|docket|content|text'), re.IGNORECASE)):
                text_blocks.append(paragraph.get_text())

            raw_text = " ".join(text_blocks)
            if not raw_text.strip():
                # Fallback: Just grab the body text if specialized classes aren't found
                raw_text = soup.body.get_text() if soup.body else ""

            return raw_text

        except Exception as e:
            print(f"[CRITICAL] Pipeline Ingestion Interrupted: {str(e)}")
            return None

    def clean_text(self, text):
        """
        Preprocesses raw legal scrapings into clean, lowercase tokenizable words.
        """
        # Remove numbers, punctuation, and system whitespaces
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def tokenize_and_pad(self, clean_text, max_block_size=256):
        """
        Translates raw text into integer tokens based on the vocabulary map,
        and pads/truncates to match the Transformer's block_size.
        """
        words = clean_text.split()
        tokens = []

        for word in words:
            # Map word to index; use <UNK> token if word is missing from the lexicon
            tokens.append(self.vocab.get(word, self.vocab["<UNK>"]))

        # Handle Truncation or Padding (Standard GPT training preparation)
        if len(tokens) > max_block_size:
            tokens = tokens[:max_block_size]
        else:
            tokens = tokens + [self.vocab["<PAD>"]] * (max_block_size - len(tokens))

        # Convert to PyTorch Tensor format matching batch specifications
        token_tensor = torch.tensor(tokens).unsqueeze(0)  # Shape: (1, block_size)
        return token_tensor


# --- Simulation of the Entire Ingestion Loop ---
if __name__ == "__main__":
    # 1. Instantiate the Pipeline
    pipeline = CREDTSScraperPipeline()

    # 2. Mock a live legal database endpoint (using a standard educational fallback for demonstration)
    mock_url = "https://www.cs.ubc.ca/~gregor/teaching/papers/intro.html"  # A dummy stable academic URL

    print("--- STEP 1: Executing Web Scraping Target ---")
    raw_data = pipeline.scrape_legal_notice(mock_url)

    if raw_data:
        print(f"[SUCCESS] Raw characters ingested: {len(raw_data)}")

        # Injecting live mock financial text in case the target URL is a clean page
        sample_legal_dark_matter = (
            "Commercial mortgage default suit filed against Metropolitan Plaza. "
            "Bank demands immediate foreclosure due to interest rate maturity wall breach. "
            "Court appoints a financial receiver to liquidate the commercial asset."
        )
        combined_text = raw_data[:100] + " " + sample_legal_dark_matter

        print("\n--- STEP 2: Cleaning Unstructured Text Data ---")
        cleaned_data = pipeline.clean_text(combined_text)
        print(f"Cleaned Segment Preview: ...{cleaned_data[-150:]}")

        print("\n--- STEP 3: Tokenizing text into Transformer Tensors ---")
        # Match the block_size = 256 we defined in the CREDTS_v2_Model
        input_tensor = pipeline.tokenize_and_pad(cleaned_data, max_block_size=256)
        print(f"Generated Tensor Shape: {input_tensor.shape} (Ready for CRE-DTS Transformer Layer)")
        print(f"First 20 Integer Tokens: {input_tensor[0][:20].tolist()}")