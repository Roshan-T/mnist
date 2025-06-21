import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

# Set page config
st.set_page_config(
    page_title="MNIST Digit Generator",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .digit-display {
        text-align: center;
        font-size: 2rem;
        color: #333;
        margin: 1rem 0;
    }
    .generate-button {
        width: 100%;
        height: 3rem;
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Generator Network Class (same as training script)
class Generator(nn.Module):
    def __init__(self, latent_dim=100, num_classes=10, img_size=28):
        super(Generator, self).__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Main generator network
        self.model = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(True),
            
            nn.Linear(1024, img_size * img_size),
            nn.Tanh()
        )
    
    def forward(self, noise, labels):
        label_emb = self.label_embedding(labels)
        gen_input = torch.cat([noise, label_emb], dim=1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, self.img_size, self.img_size)
        return img

@st.cache_resource
def load_model():
    """Load the trained generator model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Initialize the generator
        generator = Generator().to(device)
        
        # Try to load the saved model
        try:
            generator.load_state_dict(torch.load('mnist_generator.pth', map_location=device))
            st.success("✅ Model loaded successfully!")
        except FileNotFoundError:
            st.warning("⚠️ Model file 'mnist_generator.pth' not found. Using randomly initialized weights.")
            st.info("Please ensure you have the trained model file in the same directory as this script.")
        
        generator.eval()
        return generator, device
        
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None, None

def generate_digit_images(generator, device, digit, num_samples=5):
    """Generate images for a specific digit"""
    if generator is None:
        return None
    
    try:
        with torch.no_grad():
            # Create noise and labels
            noise = torch.randn(num_samples, 100).to(device)  # LATENT_DIM = 100
            labels = torch.full((num_samples,), digit, dtype=torch.long).to(device)
            
            # Generate images
            fake_images = generator(noise, labels)
            
            # Denormalize from [-1, 1] to [0, 1]
            fake_images = (fake_images + 1) / 2
            
            # Convert to numpy and move to CPU
            fake_images = fake_images.cpu().numpy()
            
            return fake_images
            
    except Exception as e:
        st.error(f"Error generating images: {str(e)}")
        return None

def create_image_grid(images):
    """Create a matplotlib figure with generated images"""
    if images is None:
        return None
    
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    fig.patch.set_facecolor('white')
    
    for i in range(5):
        axes[i].imshow(images[i, 0], cmap='gray')
        axes[i].set_title(f'Sample {i+1}', fontsize=12, pad=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    return Image.open(buf)

# Main Streamlit App
def main():
    # App header
    st.markdown('<h1 class="main-header">🔢 MNIST Digit Generator</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Generate handwritten digits using a trained GAN model</p>', unsafe_allow_html=True)
    
    # Load model
    generator, device = load_model()
    
    if generator is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Digit selection
        selected_digit = st.selectbox(
            "Select a digit to generate:",
            options=list(range(10)),
            format_func=lambda x: f"Digit {x}",
            index=0
        )
        
        st.markdown("---")
        
        # Model info
        st.subheader("📊 Model Info")
        device_info = "🖥️ CPU" if device.type == 'cpu' else "🚀 GPU"
        st.write(f"**Device:** {device_info}")
        st.write(f"**Architecture:** Conditional GAN")
        st.write(f"**Dataset:** MNIST")
        st.write(f"**Image Size:** 28x28 pixels")
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📝 How to use")
        st.write("1. Select a digit (0-9)")
        st.write("2. Click 'Generate Images'")
        st.write("3. View 5 generated samples")
        
    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f'<div class="digit-display">Selected Digit: <strong>{selected_digit}</strong></div>', unsafe_allow_html=True)
        
        # Generate button
        if st.button("🎨 Generate Images", key="generate", help="Click to generate 5 samples of the selected digit"):
            with st.spinner(f"Generating digit {selected_digit} images..."):
                # Generate images
                generated_images = generate_digit_images(generator, device, selected_digit, 5)
                
                if generated_images is not None:
                    # Create and display image grid
                    image_grid = create_image_grid(generated_images)
                    
                    if image_grid:
                        st.image(image_grid, caption=f"Generated samples of digit {selected_digit}", use_column_width=True)
                        
                        # Success message
                        st.success(f"✨ Successfully generated 5 samples of digit {selected_digit}!")
                        
                        # Download option
                        buf = io.BytesIO()
                        image_grid.save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="💾 Download Generated Images",
                            data=buf.getvalue(),
                            file_name=f"generated_digit_{selected_digit}.png",
                            mime="image/png"
                        )
                else:
                    st.error("Failed to generate images. Please try again.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #888; font-size: 0.9rem;">'
        'Generated using a Conditional GAN trained on the MNIST dataset'
        '</p>', 
        unsafe_allow_html=True
    )

# Additional features section
def show_batch_generation():
    """Show batch generation for all digits"""
    st.header("🎯 Batch Generation")
    st.write("Generate samples for all digits at once:")
    
    generator, device = load_model()
    
    if st.button("Generate All Digits", key="batch_generate"):
        with st.spinner("Generating samples for all digits..."):
            cols = st.columns(5)
            
            for digit in range(10):
                generated_images = generate_digit_images(generator, device, digit, 1)
                
                if generated_images is not None:
                    col_idx = digit % 5
                    with cols[col_idx]:
                        # Convert single image for display
                        img_array = generated_images[0, 0]
                        st.image(img_array, caption=f"Digit {digit}", use_column_width=True, clamp=True)

# Run the app
if __name__ == "__main__":
    # Add tabs for different features
    tab1, tab2 = st.tabs(["🎨 Single Digit Generator", "🎯 Batch Generator"])
    
    with tab1:
        main()
    
    with tab2:
        show_batch_generation()