import streamlit as st
import torch
import torch.nn as nn
import pickle
from PIL import Image
import torchvision.transforms as transforms
from torchvision import models
from huggingface_hub import hf_hub_download
import tempfile
import os

torch.set_num_threads(1)
# ========================================
# PAGE CONFIGURATION
# ========================================
st.set_page_config(
    page_title="Flickr30k Image Captioner",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================
# MODEL ARCHITECTURE (MUST match training)
# ========================================
class Encoder(nn.Module):
    def __init__(self, input_size=2048, hidden_size=512):
        super().__init__()
        self.fc = nn.Linear(input_size, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        return self.bn(self.fc(x))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, encoder_hidden, captions, tf_ratio=0.5):
        # Training forward pass (not used in inference)
        batch_size, seq_len = captions.size()
        outputs = torch.zeros(batch_size, seq_len, self.fc.out_features).to(encoder_hidden.device)
        input_token = captions[:, 0].unsqueeze(1)
        hidden = encoder_hidden.unsqueeze(0).contiguous()
        cell = torch.zeros_like(hidden)
        
        for t in range(1, seq_len):
            embedded = self.embedding(input_token)
            output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
            output = self.fc(output.squeeze(1))
            outputs[:, t, :] = output
            use_teacher = torch.rand(1).item() < tf_ratio
            input_token = captions[:, t].unsqueeze(1) if use_teacher else output.argmax(1).unsqueeze(1)
        
        return outputs

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, img_feats, captions, tf_ratio=0.5):
        return self.decoder(self.encoder(img_feats), captions, tf_ratio)

# ========================================
# BEAM SEARCH INFERENCE
# ========================================
def beam_search(model, img_features, vocab, max_len=30, beam_width=3, device='cuda'):
    """
    Beam search decoding for caption generation
    """
    model.eval()
    with torch.no_grad():
        # Encode image
        encoder_hidden = model.encoder(img_features.unsqueeze(0))  # (1, hidden_size)
        hidden = encoder_hidden.unsqueeze(0).contiguous()  # (1, 1, hidden_size)
        cell = torch.zeros_like(hidden)
        
        # Initialize beams
        beams = [{
            'seq': [vocab.stoi['<start>']],
            'score': 0.0,
            'hidden': hidden,
            'cell': cell
        }]
        completed_beams = []
        
        # Generate tokens
        for _ in range(max_len - 1):
            candidates = []
            
            for beam in beams:
                # Skip completed beams
                if beam['seq'][-1] == vocab.stoi['<end>']:
                    completed_beams.append(beam)
                    continue
                
                # Get next token probabilities
                input_token = torch.tensor([[beam['seq'][-1]]], device=device)
                embedded = model.decoder.embedding(input_token)
                output, (new_hidden, new_cell) = model.decoder.lstm(embedded, (beam['hidden'], beam['cell']))
                output = model.decoder.fc(output.squeeze(1))
                log_probs = torch.log_softmax(output, dim=1).squeeze()
                
                # Get top-k candidates
                topk_vals, topk_idxs = log_probs.topk(beam_width)
                for i in range(beam_width):
                    candidates.append({
                        'seq': beam['seq'] + [topk_idxs[i].item()],
                        'score': beam['score'] + topk_vals[i].item(),
                        'hidden': new_hidden,
                        'cell': new_cell
                    })
            
            # Sort by normalized score and keep top-k
            candidates.sort(key=lambda x: x['score'] / len(x['seq']), reverse=True)
            beams = candidates[:beam_width]
        
        # Add remaining beams to completed
        completed_beams.extend(beams)
        
        # Select best beam with length normalization
        if not completed_beams:
            completed_beams = beams
        
        best_beam = max(completed_beams, key=lambda x: x['score'] / len(x['seq']))
        return vocab.decode(best_beam['seq'])

# ========================================
# LOAD MODEL FUNCTION (Cached for efficiency)
# ========================================
@st.cache_resource
def load_captioning_model():
    """
    Download and load model from Hugging Face Hub
    Cached so it only downloads once per session
    """
    try:
        with st.spinner("🔄 Downloading model from Hugging Face... (one-time, ~150MB)"):
            # Download files from zentom/neural_story_teller
            model_path = hf_hub_download(
                repo_id="zentom/neural_story_teller",
                filename="best_model.pth",
                cache_dir="model_cache"
            )
            
            vocab_path = hf_hub_download(
                repo_id="zentom/neural_story_teller",
                filename="vocab.pkl",
                cache_dir="model_cache"
            )
        
        with st.spinner("🔄 Loading vocabulary..."):
            # Load vocabulary
            with open(vocab_path, 'rb') as f:
                vocab = pickle.load(f)
        
        with st.spinner("🔄 Initializing model architecture..."):
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = Seq2Seq(
                Encoder(hidden_size=512),
                Decoder(vocab_size=vocab.vocab_size, embed_size=256, hidden_size=512)
            ).to(device)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=device)
            state_dict = checkpoint['model_state_dict']
            
            # Handle DataParallel prefix if present
            if 'module.encoder.fc.weight' in state_dict:
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model.load_state_dict(state_dict)
            model.eval()
        
        with st.spinner("🔄 Loading feature extractor..."):
            # Load ResNet50 for feature extraction
            resnet = models.resnet50(weights=None)
            resnet = nn.Sequential(*list(resnet.children())[:-1])
            resnet.eval()
            resnet = resnet.to(device)
            
            # Image transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return model, vocab, resnet, transform, device
    
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.info("""
        **Troubleshooting:**
        - Check internet connection
        - Verify repo `zentom/neural_story_teller` exists and is accessible
        - Ensure files `best_model.pth` and `vocab.pkl` are uploaded
        - Check Hugging Face token permissions
        """)
        raise

# ========================================
# MAIN APP
# ========================================
def main():
    # Sidebar
    with st.sidebar:
        st.header("🖼️ About This Model")
        st.markdown("""
        **Architecture:**
        - Encoder: ResNet50 → Linear projection (2048 → 512)
        - Decoder: LSTM with embedding layer (256-dim)
        - Vocabulary: ~5,000 tokens (freq threshold = 5)
        
        **Training:**
        - Dataset: Flickr30k (31,783 images, 5 captions each)
        - Epochs: 15
        - Optimizer: Adam (lr=3e-4)
        - Loss: CrossEntropy (ignore padding)
        
        **Inference:**
        - Beam search with width = 3
        - Length-normalized scoring
        """)
        
        st.markdown("---")
        st.markdown("### 📊 Model Info")
        st.code("""
Repo: zentom/neural_story_teller
Files: best_model.pth (150MB)
       vocab.pkl (5MB)
        """)
    
    # Title
    st.title("🖼️ Flickr30k Image Captioning")
    st.markdown("### Upload an image to generate descriptive captions")
    st.markdown("""
    This model was trained on the **Flickr30k dataset** using a Seq2Seq architecture with 
    ResNet50 encoder and LSTM decoder. It generates natural language descriptions for images 
    using beam search decoding.
    """)
    
    # Load model (cached)
    try:
        model, vocab, resnet, transform, device = load_captioning_model()
        st.success(f"✅ Model loaded successfully on {device.type.upper()}")
    except Exception as e:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload any image to generate a caption"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([2, 1])
        
        with col1:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='📸 Uploaded Image', use_column_width=True)
        
        with col2:
            st.markdown("### ⚙️ Generation Settings")
            beam_width = st.slider("Beam Width", min_value=1, max_value=5, value=3, 
                                  help="Higher = better quality but slower")
            max_len = st.slider("Max Caption Length", min_value=10, max_value=50, value=30)
        
        # Generate caption
        with st.spinner("🤖 Generating caption..."):
            try:
                # Preprocess image
                img_tensor = transform(image).unsqueeze(0).to(device)
                
                # Extract features
                with torch.no_grad():
                    features = resnet(img_tensor).view(1, -1)
                
                # Generate caption using beam search
                caption = beam_search(
                    model, 
                    features[0], 
                    vocab, 
                    max_len=max_len, 
                    beam_width=beam_width, 
                    device=device
                )
                
                # Success!
                st.success("✅ Caption generated successfully!")
                st.markdown(f"### 🎯 Generated Caption:")
                st.markdown(f"#### {caption}")
                
                # Show technical details in expander
                with st.expander("🔍 Technical Details"):
                    st.markdown(f"""
                    **Model Details:**
                    - Device: {device.type.upper()}
                    - Feature dimension: 2048 → 512
                    - Vocabulary size: {vocab.vocab_size} tokens
                    - Beam width: {beam_width}
                    - Max length: {max_len} tokens
                    
                    **Processing:**
                    - Image resized to: 224×224
                    - Normalization: ImageNet mean/std
                    - Decoding method: Beam search with length normalization
                    """)
                
                # Example usage
                st.markdown("---")
                st.markdown("### 💡 Example Captions")
                st.markdown("""
                The model can generate captions like:
                - *"Two young guys with shaggy hair look at their hands while hanging out in the yard"*
                - *"Several men in hard hats are operating a giant pulley system"*
                - *"A man in a blue shirt standing in a garden"*
                """)
                
            except Exception as e:
                st.error(f"❌ Error generating caption: {str(e)}")
                st.info("Try uploading a different image or check model loading.")
    
    else:
        # Show example when no image uploaded
        st.info("👆 Upload an image to get started!")
        st.markdown("""
        ### How to Use:
        1. Click **"Choose an image..."** above
        2. Select any JPG, JPEG, or PNG file from your device
        3. Wait ~2-3 seconds for caption generation
        4. View the generated caption below the image
        
        ### Tips:
        - Use clear, well-lit images for best results
        - The model works best with common scenes (people, objects, activities)
        - Adjust beam width in sidebar for quality/speed trade-off
        """)

# ========================================
# RUN APP
# ========================================
if __name__ == "__main__":
    main()
