"""Pretrained SAM3 identity consumed by the image encoder."""

SAM3_HF_REPO_ID = "facebook/sam3"
SAM3_HF_FILENAME = "sam3.pt"
SAM3_HF_REVISION = "3c879f39826c281e95690f02c7821c4de09afae7"
SAM3_HF_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
SAM3_HF_INITIALIZATION_IDENTITY = (
    f"huggingface:{SAM3_HF_REPO_ID}:{SAM3_HF_FILENAME}@{SAM3_HF_REVISION}#{SAM3_HF_SHA256}"
)
