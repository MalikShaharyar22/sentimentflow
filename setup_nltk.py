import nltk


def download_resources():
    resources = ["punkt", "punkt_tab"]
    for res in resources:
        try:
            nltk.data.find(f"tokenizers/{res}")
            print(f"✅ {res} already installed")
        except LookupError:
            print(f"⬇️ Downloading {res} ...")
            nltk.download(res)

if __name__ == "__main__":
    download_resources()
    print("🎉 NLTK setup complete")
