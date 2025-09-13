#!/usr/bin/env python3
"""
Interactive Audio Embedding Analysis Script - 3 Audio Types
Uses Plotly for interactive visualizations with hover information showing transcripts.
Loads original human speech, original fake audio (TTS from original text), and perturbed fake audio (TTS from perturbed text)
from PWWS attack results, extracts embeddings using ReDimNet, and visualizes the distribution using PCA and t-SNE.
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import librosa
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
import hashlib
import struct
warnings.filterwarnings("ignore")

# Interactive plotting
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Dimensionality reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Set environment variable for CUDA deterministic behavior
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set deterministic behavior for PyTorch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)

class SpeechVerificationModel:
    """Wrapper for ReDimNet speech verification models"""
    
    def __init__(self, model_name: str = 'b6', dataset: str = 'vox2', train_type: str = 'ft_lm'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        self.dataset = dataset
        self.train_type = train_type
        self.model: Optional[torch.nn.Module] = None
        self.load_model()
        
    def load_model(self):
        """Load pretrained ReDimNet model"""
        try:
            print(f"Loading ReDimNet model: {self.model_name}, dataset: {self.dataset}, train_type: {self.train_type}")
            loaded_model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                                      model_name=self.model_name, 
                                      train_type=self.train_type, 
                                      dataset=self.dataset,
                                      trust_repo=True)
            if loaded_model is not None:
                self.model = loaded_model  # type: ignore
                assert self.model is not None
                self.model.to(self.device)
                self.model.eval()
                print("Model loaded successfully")
            else:
                raise ValueError("Model loading returned None")
        except Exception as e:
            print(f"Failed to load model: {e}")
            raise
    
    def extract_embedding(self, audio: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract embedding from audio using the model"""
        try:
            if self.model is None:
                print("Model is not loaded")
                return None
                
            with torch.no_grad():
                # Add batch dimension
                audio_batch = audio.unsqueeze(0).to(self.device)
                
                # Get embedding
                embedding = self.model(audio_batch)
                
                # Normalize embedding
                embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
                
                return embedding.squeeze(0).cpu()
        except Exception as e:
            print(f"Failed to extract embedding: {e}")
            return None

class InteractiveThreeTypeAudioAnalyzer:
    """
    Interactive analyzer for comparing original, original fake, and perturbed fake audio embeddings from PWWS attack results
    """
    
    def __init__(
        self, 
        json_file: str,
        original_base_dir: str = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/data/test",
        original_fake_cache_dir: str = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/cache/original-fake-audios-05072025_16khz",
        perturbed_fake_cache_dir: str = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/cache/fake_generated_audios/pwws_seed_16",
        tts_model: str = "F5TTSGenerator",
        seeding_mode: str = "seeded",
        seed_value: int = 16
    ):
        self.json_file = json_file
        self.original_base_dir = original_base_dir
        self.original_fake_cache_dir = original_fake_cache_dir
        self.perturbed_fake_cache_dir = perturbed_fake_cache_dir
        self.tts_model = tts_model
        self.seeding_mode = seeding_mode
        self.seed_value = seed_value
        
        # Load data
        self.data = self.load_json_data()
        self.sv_model = None
        self.embeddings = []
        self.metadata = []
        
    def load_json_data(self) -> Dict:
        """Load PWWS attack results from JSON file"""
        try:
            with open(self.json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Loaded {len(data['results'])} results from {self.json_file}")
            return data
        except Exception as e:
            print(f"Failed to load JSON data: {e}")
            return {}
    
    def get_text_hash(self, text: str) -> str:
        """Generate hash for text (matching tts_utils.py implementation)"""
        # Create a SHA-256 hash of the text
        hash_object = hashlib.sha256(text.encode())
        
        # Get the hash digest as bytes
        hash_bytes = hash_object.digest()
        
        # Take the first 4 bytes (32 bits) of the hash and convert them to an integer
        int32_value = struct.unpack('I', hash_bytes[:4])[0]
        
        return str(int32_value)
    
    def get_original_audio_path(self, target_audio_path: str) -> str:
        """Get the path to original human speech audio file"""
        # Original audio is directly from the target_audio_path + base directory
        return os.path.join(self.original_base_dir, target_audio_path)
    
    def get_original_fake_audio_path(self, target_audio_path: str) -> str:
        """Get the path to original fake (TTS from original text) cached audio file"""
        parts = target_audio_path.split('/')
        if len(parts) >= 3:
            speaker_id, video_id, audio_filename = parts[-3], parts[-2], parts[-1]
            
            # Add seed suffix to filename
            name, ext = os.path.splitext(audio_filename)
            seeded_filename = f"{name}_seed{self.seed_value}{ext}"
            
            cached_path = os.path.join(
                self.original_fake_cache_dir,
                self.tts_model,
                self.seeding_mode,
                speaker_id,
                video_id,
                seeded_filename
            )
            return cached_path
        return ""
    
    def get_perturbed_fake_audio_path(self, target_audio_path: str, perturbed_text: str) -> str:
        """Get the path to perturbed fake (TTS from perturbed text) cached audio file with hash-based structure"""
        parts = target_audio_path.split('/')
        if len(parts) >= 3:
            speaker_id, video_id, audio_filename = parts[-3], parts[-2], parts[-1]
            
            # Generate hash for perturbed text
            hash_value = self.get_text_hash(perturbed_text)
            
            # Create hash-based path structure
            name, ext = os.path.splitext(audio_filename)
            variant_filename = f"{hash_value}_{name}_seed{self.seed_value}{ext}"
            
            cached_path = os.path.join(
                self.perturbed_fake_cache_dir,
                self.tts_model,
                self.seeding_mode,
                speaker_id,
                video_id,
                name,
                variant_filename
            )
            return cached_path
        return ""
    
    def load_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load audio file"""
        try:
            if not os.path.exists(audio_path):
                print(f"Audio file not found: {audio_path}")
                return None
            
            audio, sr = librosa.load(audio_path, sr=16000)
            return audio
        except Exception as e:
            print(f"Failed to load audio {audio_path}: {e}")
            return None
    
    def initialize_model(self, model_name: str = 'b6', dataset: str = 'vox2', train_type: str = 'ft_lm'):
        """Initialize the speech verification model"""
        self.sv_model = SpeechVerificationModel(
            model_name=model_name,
            dataset=dataset,
            train_type=train_type
        )
    
    def extract_embeddings(self) -> bool:
        """Extract embeddings from original, original fake, and perturbed fake audio files"""
        if self.sv_model is None:
            print("Speech verification model not initialized")
            return False
        
        if not self.data or 'results' not in self.data:
            print("No data loaded")
            return False
        
        print("Extracting embeddings from original, original fake, and perturbed fake audio files...")
        
        successful_extractions = 0
        failed_extractions = 0
        sample_id = 0  # Track sample groups for connecting arrows
        
        for result in tqdm(self.data['results']):
            try:
                target_audio_path = result.get('target_audio_path', '')
                original_text = result.get('original_text', '')
                perturbed_text = result.get('perturbed_text', '')
                
                # Skip if perturbed text is same as original (no attack)
                if original_text == perturbed_text:
                    print(f"Skipping {target_audio_path} - no text perturbation")
                    continue
                
                # Get paths for all three audio types
                original_audio_path = self.get_original_audio_path(target_audio_path)
                original_fake_audio_path = self.get_original_fake_audio_path(target_audio_path)
                perturbed_fake_audio_path = self.get_perturbed_fake_audio_path(target_audio_path, perturbed_text)
                
                # Process original human speech audio
                original_audio_data = self.load_audio(original_audio_path)
                if original_audio_data is not None:
                    # Extract embedding for original audio
                    audio_tensor = torch.tensor(original_audio_data, dtype=torch.float32)
                    embedding = self.sv_model.extract_embedding(audio_tensor)
                    
                    if embedding is not None:
                        self.embeddings.append(embedding.numpy())
                        
                        # Store metadata for original audio with detailed hover info
                        metadata = {
                            'speaker_id': result.get('speaker_id', 'unknown'),
                            'video_id': result.get('video_id', 'unknown'),
                            'audio_filename': result.get('audio_filename', 'unknown'),
                            'text': original_text,
                            'audio_type': 'original',
                            'similarity_improvement': 0.0,  # N/A for original human speech
                            'original_similarity': result.get('original_similarity', 0.0),
                            'perturbed_similarity': result.get('perturbed_similarity', 0.0),
                            'target_audio_path': target_audio_path,
                            'cached_audio_path': original_audio_path,
                            'text_length': len(original_text),
                            'sample_id': sample_id,  # For connecting arrows
                            'hover_info': f"Speaker: {result.get('speaker_id', 'unknown')}<br>" +
                                        f"Type: Original Human Speech<br>" +
                                        f"Text: {original_text[:100]}{'...' if len(original_text) > 100 else ''}"
                        }
                        self.metadata.append(metadata)
                        successful_extractions += 1
                
                # Process original fake audio (TTS from original text)
                original_fake_audio_data = self.load_audio(original_fake_audio_path)
                if original_fake_audio_data is not None:
                    # Extract embedding for original fake audio
                    audio_tensor = torch.tensor(original_fake_audio_data, dtype=torch.float32)
                    embedding = self.sv_model.extract_embedding(audio_tensor)
                    
                    if embedding is not None:
                        self.embeddings.append(embedding.numpy())
                        
                        # Store metadata for original fake audio with detailed hover info
                        metadata = {
                            'speaker_id': result.get('speaker_id', 'unknown'),
                            'video_id': result.get('video_id', 'unknown'),
                            'audio_filename': result.get('audio_filename', 'unknown'),
                            'text': original_text,
                            'audio_type': 'original_fake',
                            'similarity_improvement': 0.0,  # N/A for original fake audio
                            'original_similarity': result.get('original_similarity', 0.0),
                            'perturbed_similarity': result.get('perturbed_similarity', 0.0),
                            'target_audio_path': target_audio_path,
                            'cached_audio_path': original_fake_audio_path,
                            'text_length': len(original_text),
                            'sample_id': sample_id,  # For connecting arrows
                            'hover_info': f"Speaker: {result.get('speaker_id', 'unknown')}<br>" +
                                        f"Type: Original Fake (TTS)<br>" +
                                        f"Text: {original_text[:100]}{'...' if len(original_text) > 100 else ''}<br>" +
                                        f"TTS Similarity: {result.get('original_similarity', 0.0):.3f}"
                        }
                        self.metadata.append(metadata)
                        successful_extractions += 1
                
                # Process perturbed fake audio (TTS from perturbed text)
                perturbed_fake_audio_data = self.load_audio(perturbed_fake_audio_path)
                if perturbed_fake_audio_data is not None:
                    # Extract embedding for perturbed fake audio
                    audio_tensor = torch.tensor(perturbed_fake_audio_data, dtype=torch.float32)
                    embedding = self.sv_model.extract_embedding(audio_tensor)
                    
                    if embedding is not None:
                        self.embeddings.append(embedding.numpy())
                        
                        # Store metadata for perturbed fake audio with detailed hover info
                        metadata = {
                            'speaker_id': result.get('speaker_id', 'unknown'),
                            'video_id': result.get('video_id', 'unknown'),
                            'audio_filename': result.get('audio_filename', 'unknown'),
                            'text': perturbed_text,
                            'audio_type': 'perturbed_fake',
                            'similarity_improvement': result.get('similarity_improvement', 0.0),
                            'original_similarity': result.get('original_similarity', 0.0),
                            'perturbed_similarity': result.get('perturbed_similarity', 0.0),
                            'target_audio_path': target_audio_path,
                            'cached_audio_path': perturbed_fake_audio_path,
                            'text_length': len(perturbed_text),
                            'sample_id': sample_id,  # For connecting arrows
                            'hover_info': f"Speaker: {result.get('speaker_id', 'unknown')}<br>" +
                                        f"Type: Perturbed Fake (TTS)<br>" +
                                        f"Text: {perturbed_text[:100]}{'...' if len(perturbed_text) > 100 else ''}<br>" +
                                        f"Attack Similarity: {result.get('perturbed_similarity', 0.0):.3f}"
                        }
                        self.metadata.append(metadata)
                        successful_extractions += 1
                
                if original_audio_data is None and original_fake_audio_data is None and perturbed_fake_audio_data is None:
                    failed_extractions += 1
                
                # Increment sample_id for next group of related audio files    
                sample_id += 1
                    
            except Exception as e:
                print(f"Error processing {result.get('target_audio_path', 'unknown')}: {e}")
                failed_extractions += 1
                sample_id += 1  # Still increment even on error to maintain consistency
        
        print(f"Successfully extracted {successful_extractions} embeddings")
        print(f"Failed to extract {failed_extractions} embeddings")
        
        return successful_extractions > 0
    
    def apply_dimensionality_reduction(self, n_components_pca: int = 50, perplexity: int = 50, random_state: int = 42):
        """Apply PCA and t-SNE for dimensionality reduction"""
        if not self.embeddings:
            print("No embeddings available for dimensionality reduction")
            return None, None, None
        
        print("Applying dimensionality reduction...")
        
        # Convert to numpy array
        embeddings_array = np.array(self.embeddings)
        print(f"Embeddings shape: {embeddings_array.shape}")
        
        # Standardize the embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings_array)
        
        # Apply direct 2D PCA for visualization
        print("Applying direct 2D PCA...")
        pca_2d = PCA(n_components=2, random_state=random_state)
        embeddings_pca_2d = pca_2d.fit_transform(embeddings_scaled)
        
        print(f"Direct 2D PCA explained variance ratio: {pca_2d.explained_variance_ratio_}")
        print(f"Total explained variance (2D PCA): {pca_2d.explained_variance_ratio_.sum():.3f}")
        
        # Apply PCA first to reduce to manageable dimensions (existing functionality)
        print(f"Applying PCA to reduce to {n_components_pca} components...")
        pca = PCA(n_components=n_components_pca, random_state=random_state)
        embeddings_pca = pca.fit_transform(embeddings_scaled)
        
        print(f"PCA explained variance ratio (first 10 components): {pca.explained_variance_ratio_[:10]}")
        print(f"Total explained variance (first {n_components_pca} components): {pca.explained_variance_ratio_.sum():.3f}")
        
        # Apply t-SNE for 2D visualization
        print("Applying t-SNE for 2D visualization...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, 
                   init='pca', learning_rate='auto')
        embeddings_tsne = tsne.fit_transform(embeddings_pca)
        
        return embeddings_pca, embeddings_tsne, embeddings_pca_2d
    
    def add_connection_arrows(self, fig, df, x_col, y_col, speakers, speaker_color_map):
        """Add arrow traces connecting human speech -> original fake -> perturbed fake for each sample"""
        # Group by sample_id to connect related audio types
        for sample_id in df['sample_id'].unique():
            sample_data = df[df['sample_id'] == sample_id]
            
            # Get speaker for this sample (should be same for all three audio types)
            speaker = sample_data['speaker_id'].iloc[0]
            speaker_color = speaker_color_map[speaker]
            
            # Get coordinates for each audio type in this sample
            original_data = sample_data[sample_data['audio_type'] == 'original']
            original_fake_data = sample_data[sample_data['audio_type'] == 'original_fake']
            perturbed_fake_data = sample_data[sample_data['audio_type'] == 'perturbed_fake']
            
            # Create arrow from human speech to original fake TTS
            if len(original_data) > 0 and len(original_fake_data) > 0:
                x0, y0 = original_data[x_col].iloc[0], original_data[y_col].iloc[0]
                x1, y1 = original_fake_data[x_col].iloc[0], original_fake_data[y_col].iloc[0]
                
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines+markers',
                    line=dict(color='gray', width=1),
                    marker=dict(
                        size=[0, 6],  # No marker at start, arrowhead at end
                        symbol=['circle', 'triangle-right'],
                        color='gray'
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                    legendgroup=speaker,  # Group with speaker for visibility control
                    name=f"{speaker} (Arrow 1)"  # Hidden name for arrow trace
                ))
            
            # Create arrow from original fake TTS to perturbed fake TTS
            if len(original_fake_data) > 0 and len(perturbed_fake_data) > 0:
                x0, y0 = original_fake_data[x_col].iloc[0], original_fake_data[y_col].iloc[0]
                x1, y1 = perturbed_fake_data[x_col].iloc[0], perturbed_fake_data[y_col].iloc[0]
                
                fig.add_trace(go.Scatter(
                    x=[x0, x1],
                    y=[y0, y1],
                    mode='lines+markers',
                    line=dict(color='red', width=1),
                    marker=dict(
                        size=[0, 6],  # No marker at start, arrowhead at end
                        symbol=['circle', 'triangle-right'],
                        color='red'
                    ),
                    hoverinfo='skip',
                    showlegend=False,
                    legendgroup=speaker,  # Group with speaker for visibility control
                    name=f"{speaker} (Arrow 2)"  # Hidden name for arrow trace
                ))
    
    def create_interactive_visualizations(self, embeddings_pca, embeddings_tsne, embeddings_pca_2d, output_dir: str = "interactive_analysis_results"):
        """Create and save interactive Plotly visualizations"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame for plotting
        df = pd.DataFrame(self.metadata)
        df['pca_1'] = embeddings_pca[:, 0]
        df['pca_2'] = embeddings_pca[:, 1]
        df['tsne_1'] = embeddings_tsne[:, 0]
        df['tsne_2'] = embeddings_tsne[:, 1]
        df['pca_2d_1'] = embeddings_pca_2d[:, 0]
        df['pca_2d_2'] = embeddings_pca_2d[:, 1]
        
        # Define color mapping for speakers with enough unique colors
        speakers = sorted(df['speaker_id'].unique())  # Sort for consistent ordering
        
        # Use multiple color palettes to ensure enough unique colors
        all_colors = (
            px.colors.qualitative.Set1 + 
            px.colors.qualitative.Set2 + 
            px.colors.qualitative.Set3 + 
            px.colors.qualitative.Pastel1 + 
            px.colors.qualitative.Pastel2
        )
        
        # Ensure we have enough colors for all speakers
        if len(speakers) > len(all_colors):
            # Generate additional colors using a simple color cycle
            import colorsys
            additional_colors = []
            for i in range(len(speakers) - len(all_colors)):
                hue = (i * 0.618033988749895) % 1  # Golden ratio for even distribution
                rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
                hex_color = '#{:02x}{:02x}{:02x}'.format(int(rgb[0]*255), int(rgb[1]*255), int(rgb[2]*255))
                additional_colors.append(hex_color)
            all_colors.extend(additional_colors)
        
        # Create color mapping ensuring every speaker gets a unique color
        speaker_color_map = {speaker: all_colors[i % len(all_colors)] for i, speaker in enumerate(speakers)}
        
        print(f"Assigned unique colors to {len(speakers)} speakers")
        
        # Add color column for speakers
        df['speaker_color'] = df['speaker_id'].apply(lambda x: speaker_color_map.get(x, '#000000'))
        
        # Define audio type symbols
        audio_type_symbols = {
            'original': 'circle',
            'original_fake': 'square', 
            'perturbed_fake': 'triangle-up'
        }
        
        # 1. Interactive t-SNE plot: Speaker colors with audio type shapes
        fig1 = go.Figure()
        
        for speaker in speakers:
            speaker_data = df[df['speaker_id'] == speaker]
            speaker_color = speaker_color_map[speaker]
            
            for audio_type in ['original', 'original_fake', 'perturbed_fake']:
                type_data = speaker_data[speaker_data['audio_type'] == audio_type]
                if len(type_data) > 0:
                    # Create name showing both speaker and type
                    type_name_map = {
                        'original': 'Human',
                        'original_fake': 'Orig TTS', 
                        'perturbed_fake': 'Pert TTS'
                    }
                    trace_name = f"{speaker} ({type_name_map[audio_type]})"
                    
                    fig1.add_trace(go.Scatter(
                        x=type_data['tsne_1'],
                        y=type_data['tsne_2'],
                        mode='markers',
                        marker=dict(
                            color=speaker_color,
                            size=8,
                            symbol=audio_type_symbols[audio_type],
                            line=dict(width=1, color='white')
                        ),
                        name=trace_name,
                        text=type_data['text'],
                        customdata=type_data['hover_info'],
                        hovertemplate='<b>%{customdata}</b><br>' +
                                     'Full Text: %{text}<br>' +
                                     't-SNE 1: %{x:.2f}<br>' +
                                     't-SNE 2: %{y:.2f}<extra></extra>',
                        legendgroup=speaker,
                        showlegend=True
                    ))
        
        # Add connection arrows showing progression from human speech to original fake to perturbed fake
        self.add_connection_arrows(fig1, df, 'tsne_1', 'tsne_2', speakers, speaker_color_map)
        
        # Update layout with buttons AFTER arrows are added so button logic works correctly
        fig1.update_layout(
            title='Interactive t-SNE: Speaker Colors with Audio Type Shapes (Combined 6 Batches)<br><sub>Circle=Original Human, Square=Original Fake(TTS), Triangle=Perturbed Fake(TTS) | Each Speaker has Unique Color<br>Gray arrows: Human→Original TTS, Red arrows: Original TTS→Perturbed TTS</sub>',
            xaxis_title='t-SNE Component 1',
            yaxis_title='t-SNE Component 2',
            width=1200,
            height=800,
            hovermode='closest',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig1.data)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": ["legendonly"] * len(fig1.data)}],
                            label="Hide All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False if "Arrow" in trace.name else True for trace in fig1.data]}],
                            label="Hide Arrows",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [True] * len(fig1.data)}],
                            label="Show Arrows",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        fig1.write_html(os.path.join(output_dir, 'interactive_tsne_three_types.html'))
        print(f"Saved interactive t-SNE plot to {output_dir}/interactive_tsne_three_types.html")
        
        # 2. Interactive PCA plot: Speaker colors with audio type shapes
        fig2 = go.Figure()
        
        for speaker in speakers:
            speaker_data = df[df['speaker_id'] == speaker]
            speaker_color = speaker_color_map[speaker]
            
            for audio_type in ['original', 'original_fake', 'perturbed_fake']:
                type_data = speaker_data[speaker_data['audio_type'] == audio_type]
                if len(type_data) > 0:
                    # Create name showing both speaker and type
                    type_name_map = {
                        'original': 'Human',
                        'original_fake': 'Orig TTS', 
                        'perturbed_fake': 'Pert TTS'
                    }
                    trace_name = f"{speaker} ({type_name_map[audio_type]})"
                    
                    fig2.add_trace(go.Scatter(
                        x=type_data['pca_1'],
                        y=type_data['pca_2'],
                        mode='markers',
                        marker=dict(
                            color=speaker_color,
                            size=8,
                            symbol=audio_type_symbols[audio_type],
                            line=dict(width=1, color='white')
                        ),
                        name=trace_name,
                        text=type_data['text'],
                        customdata=type_data['hover_info'],
                        hovertemplate='<b>%{customdata}</b><br>' +
                                     'Full Text: %{text}<br>' +
                                     'PCA 1: %{x:.2f}<br>' +
                                     'PCA 2: %{y:.2f}<extra></extra>',
                        legendgroup=speaker,
                        showlegend=True
                    ))
        
        # Add connection arrows showing progression from human speech to original fake to perturbed fake
        self.add_connection_arrows(fig2, df, 'pca_1', 'pca_2', speakers, speaker_color_map)
        
        # Update layout with buttons AFTER arrows are added so button logic works correctly
        fig2.update_layout(
            title='Interactive PCA: Speaker Colors with Audio Type Shapes (Combined 6 Batches)<br><sub>Circle=Original Human, Square=Original Fake(TTS), Triangle=Perturbed Fake(TTS) | Each Speaker has Unique Color<br>Gray arrows: Human→Original TTS, Red arrows: Original TTS→Perturbed TTS</sub>',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            width=1200,
            height=800,
            hovermode='closest',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig2.data)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": ["legendonly"] * len(fig2.data)}],
                            label="Hide All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False if "Arrow" in trace.name else True for trace in fig2.data]}],
                            label="Hide Arrows",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [True] * len(fig2.data)}],
                            label="Show Arrows",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        fig2.write_html(os.path.join(output_dir, 'interactive_pca_three_types.html'))
        print(f"Saved interactive PCA plot to {output_dir}/interactive_pca_three_types.html")
        
        # 3. Interactive similarity analysis with speaker colors
        fig3 = px.scatter(
            df, 
            x='original_similarity', 
            y='perturbed_similarity',
            color='speaker_id',
            symbol='audio_type',
            size='text_length',
            hover_data=['audio_type'],
            title='Interactive Similarity Analysis: Speaker Colors with Audio Type Shapes (Combined 6 Batches)',
            labels={
                'original_similarity': 'Original Similarity',
                'perturbed_similarity': 'Perturbed Similarity',
                'speaker_id': 'Speaker',
                'audio_type': 'Audio Type'
            },
            custom_data=['text']
        )
        
        fig3.update_traces(
            hovertemplate='<b>%{customdata[0]}</b><br>' +
                         'Speaker: %{color}<br>' +
                         'Type: %{customdata[1]}<br>' +
                         'Original Sim: %{x:.3f}<br>' +
                         'Perturbed Sim: %{y:.3f}<br>' +
                         'Text Length: %{marker.size}<extra></extra>'
        )
        
        # Add diagonal line
        fig3.add_shape(
            type="line",
            x0=df['original_similarity'].min(),
            y0=df['original_similarity'].min(),
            x1=df['original_similarity'].max(),
            y1=df['original_similarity'].max(),
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig3.update_layout(
            width=1200, 
            height=800,
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig3.data)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": ["legendonly"] * len(fig3.data)}],
                            label="Hide All",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        fig3.write_html(os.path.join(output_dir, 'interactive_similarity_analysis_three_types.html'))
        print(f"Saved interactive similarity plot to {output_dir}/interactive_similarity_analysis_three_types.html")
        
        # 4. Interactive direct 2D PCA plot: Speaker colors with audio type shapes
        fig4 = go.Figure()
        
        for speaker in speakers:
            speaker_data = df[df['speaker_id'] == speaker]
            speaker_color = speaker_color_map[speaker]
            
            for audio_type in ['original', 'original_fake', 'perturbed_fake']:
                type_data = speaker_data[speaker_data['audio_type'] == audio_type]
                if len(type_data) > 0:
                    # Create name showing both speaker and type
                    type_name_map = {
                        'original': 'Human',
                        'original_fake': 'Orig TTS', 
                        'perturbed_fake': 'Pert TTS'
                    }
                    trace_name = f"{speaker} ({type_name_map[audio_type]})"
                    
                    fig4.add_trace(go.Scatter(
                        x=type_data['pca_2d_1'],
                        y=type_data['pca_2d_2'],
                        mode='markers',
                        marker=dict(
                            color=speaker_color,
                            size=8,
                            symbol=audio_type_symbols[audio_type],
                            line=dict(width=1, color='white')
                        ),
                        name=trace_name,
                        text=type_data['text'],
                        customdata=type_data['hover_info'],
                        hovertemplate='<b>%{customdata}</b><br>' +
                                     'Full Text: %{text}<br>' +
                                     'PCA 2D 1: %{x:.2f}<br>' +
                                     'PCA 2D 2: %{y:.2f}<extra></extra>',
                        legendgroup=speaker,
                        showlegend=True
                    ))
        
        # Add connection arrows showing progression from human speech to original fake to perturbed fake
        self.add_connection_arrows(fig4, df, 'pca_2d_1', 'pca_2d_2', speakers, speaker_color_map)
        
        # Update layout with buttons AFTER arrows are added so button logic works correctly
        fig4.update_layout(
            title='Interactive Direct 2D PCA: Speaker Colors with Audio Type Shapes (Combined 6 Batches)<br><sub>Circle=Original Human, Square=Original Fake(TTS), Triangle=Perturbed Fake(TTS) | Each Speaker has Unique Color<br>Gray arrows: Human→Original TTS, Red arrows: Original TTS→Perturbed TTS</sub>',
            xaxis_title='First Principal Component',
            yaxis_title='Second Principal Component',
            width=1200,
            height=800,
            hovermode='closest',
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(
                            args=[{"visible": [True] * len(fig4.data)}],
                            label="Show All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": ["legendonly"] * len(fig4.data)}],
                            label="Hide All",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [False if "Arrow" in trace.name else True for trace in fig4.data]}],
                            label="Hide Arrows",
                            method="restyle"
                        ),
                        dict(
                            args=[{"visible": [True] * len(fig4.data)}],
                            label="Show Arrows",
                            method="restyle"
                        )
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        fig4.write_html(os.path.join(output_dir, 'interactive_direct_pca_three_types.html'))
        print(f"Saved interactive direct PCA plot to {output_dir}/interactive_direct_pca_three_types.html")
        
        # Save the DataFrame for further analysis
        df.to_csv(os.path.join(output_dir, 'interactive_analysis_data_three_types.csv'), index=False)
        print(f"Data saved to {output_dir}/interactive_analysis_data_three_types.csv")
        
        return df, fig1, fig2, fig3, fig4
    
    def save_static_plots(self, fig1, fig2, fig3, fig4, output_dir: str):
        """Save static versions of plots in multiple formats for sharing"""
        try:
            import kaleido  # For static image export
            
            print("Saving static plots in multiple formats...")
            
            # Create static subdirectory
            static_dir = os.path.join(output_dir, "static_exports")
            os.makedirs(static_dir, exist_ok=True)
            
            # Configure plots for static export (larger size, better quality)
            static_config = dict(width=1400, height=1000, scale=2)
            
            # Save t-SNE plot
            fig1.write_image(os.path.join(static_dir, "tsne_three_types.png"), **static_config)
            fig1.write_image(os.path.join(static_dir, "tsne_three_types.svg"), **static_config)
            fig1.write_image(os.path.join(static_dir, "tsne_three_types.pdf"), **static_config)
            
            # Save PCA plot  
            fig2.write_image(os.path.join(static_dir, "pca_three_types.png"), **static_config)
            fig2.write_image(os.path.join(static_dir, "pca_three_types.svg"), **static_config)
            fig2.write_image(os.path.join(static_dir, "pca_three_types.pdf"), **static_config)
            
            # Save similarity plot
            fig3.write_image(os.path.join(static_dir, "similarity_analysis_three_types.png"), **static_config)
            fig3.write_image(os.path.join(static_dir, "similarity_analysis_three_types.svg"), **static_config)
            fig3.write_image(os.path.join(static_dir, "similarity_analysis_three_types.pdf"), **static_config)
            
            # Save direct 2D PCA plot
            fig4.write_image(os.path.join(static_dir, "direct_pca_three_types.png"), **static_config)
            fig4.write_image(os.path.join(static_dir, "direct_pca_three_types.svg"), **static_config)
            fig4.write_image(os.path.join(static_dir, "direct_pca_three_types.pdf"), **static_config)
            
            print(f"Static plots saved to {static_dir}/")
            print("Available formats: PNG (for presentations), SVG (vector/scalable), PDF (publications)")
            
        except ImportError:
            print("Warning: kaleido not installed. Static image export skipped.")
            print("To enable static exports, install: pip install kaleido")
        except Exception as e:
            print(f"Error saving static plots: {e}")
    
    def save_data_exports(self, df, output_dir: str):
        """Save data in multiple formats for analysis in other tools"""
        try:
            # Create data subdirectory
            data_dir = os.path.join(output_dir, "data_exports")
            os.makedirs(data_dir, exist_ok=True)
            
            # Save as CSV (Excel compatible)
            df.to_csv(os.path.join(data_dir, "audio_analysis_data.csv"), index=False)
            
            # Save as JSON (for web applications)
            df.to_json(os.path.join(data_dir, "audio_analysis_data.json"), orient='records', indent=2)
            
            # Save embeddings separately for ML analysis
            embeddings_df = df[['speaker_id', 'audio_type', 'sample_id', 'pca_1', 'pca_2', 'tsne_1', 'tsne_2', 'pca_2d_1', 'pca_2d_2']].copy()
            embeddings_df.to_csv(os.path.join(data_dir, "embeddings_coordinates.csv"), index=False)
            
            # Save summary statistics
            summary_stats = {
                'total_samples': len(df),
                'unique_speakers': len(df['speaker_id'].unique()),
                'audio_type_counts': df['audio_type'].value_counts().to_dict(),
                'speaker_list': sorted(df['speaker_id'].unique()),
                'similarity_stats': {
                    'original_sim_mean': df['original_similarity'].mean(),
                    'original_sim_std': df['original_similarity'].std(),
                    'perturbed_sim_mean': df['perturbed_similarity'].mean(),
                    'perturbed_sim_std': df['perturbed_similarity'].std()
                }
            }
            
            import json
            with open(os.path.join(data_dir, "analysis_summary.json"), 'w') as f:
                json.dump(summary_stats, f, indent=2, default=str)
            
            print(f"Data exports saved to {data_dir}/")
            print("Available formats: CSV (Excel), JSON (web), summary statistics")
            
        except Exception as e:
            print(f"Error saving data exports: {e}")
    
    def run_analysis(self, model_name: str = 'b6', dataset: str = 'vox2', train_type: str = 'ft_lm', 
                    output_dir: str = "interactive_analysis_results"):
        """Run the complete interactive three-type audio embedding analysis pipeline"""
        print("Starting interactive three-type audio embedding analysis...")
        
        # Initialize model
        self.initialize_model(model_name, dataset, train_type)
        
        # Extract embeddings
        if not self.extract_embeddings():
            print("Failed to extract embeddings")
            return None
        
        # Apply dimensionality reduction
        embeddings_pca, embeddings_tsne, embeddings_pca_2d = self.apply_dimensionality_reduction()
        if embeddings_pca is None or embeddings_tsne is None or embeddings_pca_2d is None:
            print("Failed to apply dimensionality reduction")
            return None
        
        # Create interactive visualizations
        df, fig1, fig2, fig3, fig4 = self.create_interactive_visualizations(embeddings_pca, embeddings_tsne, embeddings_pca_2d, output_dir)
        
        # Save static versions for sharing
        self.save_static_plots(fig1, fig2, fig3, fig4, output_dir)
        
        # Save data in multiple formats
        self.save_data_exports(df, output_dir)
        
        # Print summary statistics
        print("\n" + "="*60)
        print("INTERACTIVE THREE-TYPE AUDIO ANALYSIS SUMMARY")
        print("="*60)
        print(f"Total samples analyzed: {len(self.embeddings)}")
        print(f"Original human speech samples: {len(df[df['audio_type'] == 'original'])}")
        print(f"Original fake (TTS) samples: {len(df[df['audio_type'] == 'original_fake'])}")
        print(f"Perturbed fake (TTS) samples: {len(df[df['audio_type'] == 'perturbed_fake'])}")
        print(f"Number of unique speakers: {len(df['speaker_id'].unique())}")
        
        # Type-wise statistics
        print("\nAudio Type Statistics:")
        type_stats = df.groupby('audio_type').agg({
            'original_similarity': ['count', 'mean', 'std'],
            'perturbed_similarity': ['mean', 'std']
        })
        print(type_stats)
        
        print(f"\nInteractive visualizations saved to {output_dir}/")
        print("Open the HTML files in a web browser to explore the data interactively!")
        
        return df

def main():
    # Configuration - Using combined dataset from all batches
    json_file = "real_attack_results/pwws_vox1-O_b6_vox2_ft_lm_16.json"
    original_base_dir = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/data/test"
    original_fake_cache_dir = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/cache/original-fake-audios-05072025_16khz"
    perturbed_fake_cache_dir = "/media/volume/AudioUnlearnData1/emnlp_audio_adversarial/cache/fake_generated_audios/pwws_seed_16"
    output_dir = "interactive_analysis_results_three_types_1807"
    
    # Create analyzer
    analyzer = InteractiveThreeTypeAudioAnalyzer(
        json_file=json_file,
        original_base_dir=original_base_dir,
        original_fake_cache_dir=original_fake_cache_dir,
        perturbed_fake_cache_dir=perturbed_fake_cache_dir,
        tts_model="F5TTSGenerator",
        seeding_mode="seeded",
        seed_value=16
    )
    
    # Run analysis
    df = analyzer.run_analysis(
        model_name='b6',
        dataset='vox2', 
        train_type='ft_lm',
        output_dir=output_dir
    )
    
    if df is not None:
        print(f"\nThree-type interactive analysis completed successfully!")
        print(f"Analyzed {len(df)} total samples from {len(df['speaker_id'].unique())} speakers")
        print(f"Combined results from 6 batches saved to {output_dir}/")
        print("\nGenerated files:")
        print(f"Interactive (HTML):")
        print(f"- {output_dir}/interactive_tsne_three_types.html")
        print(f"- {output_dir}/interactive_pca_three_types.html") 
        print(f"- {output_dir}/interactive_direct_pca_three_types.html")
        print(f"- {output_dir}/interactive_similarity_analysis_three_types.html")
        print(f"\nStatic exports (for sharing):")
        print(f"- {output_dir}/static_exports/ (PNG, SVG, PDF versions)")
        print(f"\nData exports:")
        print(f"- {output_dir}/data_exports/ (CSV, JSON, summary files)")
    else:
        print("Three-type interactive analysis failed!")

if __name__ == "__main__":
    main() 