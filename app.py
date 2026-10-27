"""ChangeMap IR-MAD - Streamlit Application"""

import streamlit as st
import numpy as np
from pathlib import Path
import pandas as pd
import yaml
from PIL import Image

# Import project modules
from src.data.dataset_loader import LevirDataset, SysuDataset, load_image_pair
from src.detection.pipelines import get_pipeline
from src.evaluation.metrics import calculate_metrics
from src.visualization.plotting import create_overlay, plot_comparison, plot_metrics
from src.preprocessing.image_utils import normalize_image

# Page configuration
st.set_page_config(
    page_title="ChangeMap IR-MAD",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load configuration
def load_config():
    config_path = Path("config/settings.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return None

config = load_config()

# Initialize session state
if 'dataset' not in st.session_state:
    st.session_state.dataset = None
if 'selected_pair' not in st.session_state:
    st.session_state.selected_pair = None
if 'pipeline_results' not in st.session_state:
    st.session_state.pipeline_results = {}

# Sidebar
st.sidebar.title("ChangeMap IR-MAD")
st.sidebar.markdown("Classical Change Detection in Satellite Imagery")

page = st.sidebar.radio(
    "Select Page",
    ["Dataset Explorer", "Change Detection", "Algorithm Comparison"]
)

# Page 1: Dataset Explorer
if page == "Dataset Explorer":
    st.title("📁 Dataset Explorer")
    st.markdown("Explore LEVIR-CD and SYSU-CD datasets")
    
    # Dataset selection
    dataset_name = st.selectbox("Select Dataset", ["LEVIR-CD", "SYSU-CD"])
    
    try:
        if dataset_name == "LEVIR-CD":
            dataset = LevirDataset()
        else:
            dataset = SysuDataset()
        
        st.session_state.dataset = dataset
        
        # Display dataset info
        col1, col2, col3 = st.columns(3)
        col1.metric("Dataset Name", dataset.dataset_name)
        col2.metric("Total Pairs", len(dataset))
        col3.metric("Splits", ", ".join(dataset.get_split_names()))
        
        st.markdown(f"**Description:** {dataset.description}")
        
        # Select split
        splits = dataset.get_split_names()
        if splits:
            selected_split = st.selectbox("Select Split", splits)
            pairs = dataset.get_image_pairs_by_split(selected_split)
        else:
            pairs = dataset.get_image_pairs()
        
        st.markdown(f"**Available pairs in this split:** {len(pairs)}")
        
        # Select image pair
        if pairs:
            pair_names = [Path(p[0]).stem for p in pairs]
            selected_pair_idx = st.selectbox("Select Image Pair", range(len(pairs)), format_func=lambda x: pair_names[x])
            
            earlier_path, later_path, label_path = pairs[selected_pair_idx]
            
            # Load images
            earlier_img, later_img, label_img = load_image_pair(
                earlier_path, later_path, label_path, root_path=dataset.root_path
            )
            
            # Display images
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(earlier_img, caption="Earlier Image", use_column_width=True)
                st.caption(f"Dimensions: {earlier_img.shape}")
            with col2:
                st.image(later_img, caption="Later Image", use_column_width=True)
                st.caption(f"Dimensions: {later_img.shape}")
            with col3:
                # Scale binary mask to 0-255 for proper visualization
                label_display = (label_img * 255).astype(np.uint8)
                st.image(label_display, caption="Ground Truth Mask", use_column_width=True)
                st.caption(f"Dimensions: {label_img.shape}")
            
            # Store selected pair for other pages
            st.session_state.selected_pair = {
                'dataset': dataset,
                'earlier_path': earlier_path,
                'later_path': later_path,
                'label_path': label_path,
                'earlier_img': earlier_img,
                'later_img': later_img,
                'label_img': label_img
            }
        else:
            st.warning("No image pairs found in this split.")
    
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Please ensure the dataset directories exist and contain the required structure.")

# Page 2: Change Detection
elif page == "Change Detection":
    st.title("🔍 Change Detection")
    st.markdown("Run change detection algorithms on image pairs")
    
    # Check if dataset is loaded
    if st.session_state.dataset is None:
        st.warning("Please select a dataset and image pair from the Dataset Explorer first.")
        st.stop()
    
    # Option to use selected pair or upload custom images
    use_custom = st.checkbox("Upload Custom Images", value=False)
    
    if use_custom:
        col1, col2 = st.columns(2)
        with col1:
            earlier_upload = st.file_uploader("Upload Earlier Image", type=['png', 'jpg', 'jpeg'])
        with col2:
            later_upload = st.file_uploader("Upload Later Image", type=['png', 'jpg', 'jpeg'])
        
        if earlier_upload and later_upload:
            earlier_img = np.array(Image.open(earlier_upload))
            later_img = np.array(Image.open(later_upload))
            label_img = None
            st.success("Custom images loaded successfully!")
        else:
            st.info("Please upload both images.")
            st.stop()
    else:
        if st.session_state.selected_pair is None:
            st.warning("Please select an image pair from the Dataset Explorer first.")
            st.stop()
        
        earlier_img = st.session_state.selected_pair['earlier_img']
        later_img = st.session_state.selected_pair['later_img']
        label_img = st.session_state.selected_pair['label_img']
    
    # Pipeline selection
    pipeline_name = st.selectbox("Select Pipeline", ["Baseline", "CVA", "IR-MAD"])
    
    # Pipeline parameters
    st.subheader("Pipeline Parameters")
    
    if pipeline_name == "Baseline":
        col1, col2 = st.columns(2)
        with col1:
            use_filter = st.checkbox("Use Filter", value=True)
            filter_type = st.selectbox("Filter Type", ["gaussian", "median"])
            filter_size = st.slider("Filter Size", 3, 11, 5, 2)
        with col2:
            threshold_method = st.selectbox("Threshold Method", ["otsu", "fixed"])
            fixed_threshold = st.slider("Fixed Threshold", 0.0, 1.0, 0.5, 0.05)
            morph_kernel_size = st.slider("Morphology Kernel Size", 3, 9, 3, 2)
        
        params = {
            'use_filter': use_filter,
            'filter_type': filter_type,
            'filter_size': filter_size,
            'threshold_method': threshold_method,
            'fixed_threshold_value': fixed_threshold,
            'morph_kernel_size': morph_kernel_size
        }
    
    elif pipeline_name == "CVA":
        col1, col2 = st.columns(2)
        with col1:
            threshold_method = st.selectbox("Threshold Method", ["otsu", "fixed"])
            fixed_threshold = st.slider("Fixed Threshold", 0.0, 1.0, 0.3, 0.05)
        with col2:
            use_morphology = st.checkbox("Use Morphology", value=True)
            morph_kernel_size = st.slider("Morphology Kernel Size", 3, 9, 3, 2)
        
        params = {
            'threshold_method': threshold_method,
            'fixed_threshold_value': fixed_threshold,
            'use_morphology': use_morphology,
            'morph_kernel_size': morph_kernel_size
        }
    
    elif pipeline_name == "IR-MAD":
        col1, col2 = st.columns(2)
        with col1:
            max_iterations = st.slider("Max Iterations", 10, 100, 30, 5)
            convergence_threshold = st.slider("Convergence Threshold", 1e-8, 1e-4, 1e-6, format="%.1e")
        with col2:
            threshold_method = st.selectbox("Threshold Method", ["otsu", "fixed"])
            fixed_threshold = st.slider("Fixed Threshold", 0.0, 1.0, 0.95, 0.05)
            use_morphology = st.checkbox("Use Morphology", value=True)
            morph_kernel_size = st.slider("Morphology Kernel Size", 3, 9, 3, 2)
        
        params = {
            'max_iterations': max_iterations,
            'convergence_threshold': convergence_threshold,
            'threshold_method': threshold_method,
            'fixed_threshold_value': fixed_threshold,
            'use_morphology': use_morphology,
            'morph_kernel_size': morph_kernel_size
        }
    
    # Run pipeline
    if st.button("Run Change Detection", type="primary"):
        with st.spinner("Running change detection..."):
            try:
                pipeline = get_pipeline(pipeline_name)
                predicted_mask, intermediate = pipeline.run(earlier_img, later_img, **params)
                
                # Store results
                st.session_state.pipeline_results[pipeline_name] = {
                    'predicted_mask': predicted_mask,
                    'intermediate': intermediate,
                    'params': params
                }
                
                st.success("Change detection completed!")
                
                # Display results
                st.subheader("Results")
                
                # Get difference/change image
                if 'difference' in intermediate:
                    diff_img = intermediate['difference']
                elif 'change_magnitude' in intermediate:
                    diff_img = intermediate['change_magnitude']
                elif 'change_probability' in intermediate:
                    diff_img = intermediate['change_probability']
                else:
                    diff_img = np.zeros_like(earlier_img[:, :, 0])
                
                # Create overlay
                overlay = create_overlay(later_img, predicted_mask, color=(255, 0, 0), alpha=0.5)
                
                # Display images
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(earlier_img, caption="Earlier Image", use_column_width=True)
                with col2:
                    st.image(later_img, caption="Later Image", use_column_width=True)
                with col3:
                    st.image(diff_img, caption=f"{pipeline_name} Change Image", clamp=True, use_column_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.image(predicted_mask * 255, caption="Predicted Mask", clamp=True, use_column_width=True)
                with col2:
                    st.image(overlay, caption="Overlay", use_column_width=True)
                with col3:
                    if label_img is not None:
                        st.image(label_img * 255, caption="Ground Truth", clamp=True, use_column_width=True)
                    else:
                        st.info("No ground truth available")
                
                # Calculate metrics if ground truth is available
                if label_img is not None:
                    st.subheader("Evaluation Metrics")
                    metrics = calculate_metrics(predicted_mask, label_img)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Precision", f"{metrics.precision:.4f}")
                    col2.metric("Recall", f"{metrics.recall:.4f}")
                    col3.metric("F1-Score", f"{metrics.f1:.4f}")
                    col4.metric("IoU", f"{metrics.iou:.4f}")
                    
                    # Store metrics for export
                    st.session_state.pipeline_results[pipeline_name]['metrics'] = metrics.to_dict()
            
            except Exception as e:
                st.error(f"Error running pipeline: {str(e)}")
                import traceback
                st.text(traceback.format_exc())

# Page 3: Algorithm Comparison
elif page == "Algorithm Comparison":
    st.title("📊 Algorithm Comparison")
    st.markdown("Compare the three pipelines on a dataset subset")
    
    # Dataset selection
    dataset_name = st.selectbox("Select Dataset", ["LEVIR-CD", "SYSU-CD"])
    
    try:
        if dataset_name == "LEVIR-CD":
            dataset = LevirDataset()
        else:
            dataset = SysuDataset()
        
        # Select split
        splits = dataset.get_split_names()
        if splits:
            selected_split = st.selectbox("Select Split", splits)
            pairs = dataset.get_image_pairs_by_split(selected_split)
        else:
            pairs = dataset.get_image_pairs()
        
        # Select number of images to evaluate
        max_images = min(len(pairs), 10)
        num_images = st.slider("Number of images to evaluate", 1, max_images, min(5, max_images))
        
        # Select specific images
        pair_names = [Path(p[0]).stem for p in pairs]
        selected_indices = st.multiselect(
            "Select images to evaluate",
            range(len(pairs)),
            default=list(range(min(num_images, len(pairs)))),
            format_func=lambda x: pair_names[x]
        )
        
        if not selected_indices:
            st.warning("Please select at least one image to evaluate.")
            st.stop()
        
        # Run comparison
        if st.button("Run Comparison", type="primary"):
            with st.spinner("Running comparison..."):
                results = {
                    'Baseline': {'precision': [], 'recall': [], 'f1': [], 'iou': []},
                    'CVA': {'precision': [], 'recall': [], 'f1': [], 'iou': []},
                    'IR-MAD': {'precision': [], 'recall': [], 'f1': [], 'iou': []}
                }
                
                for idx in selected_indices:
                    earlier_path, later_path, label_path = pairs[idx]
                    earlier_img, later_img, label_img = load_image_pair(
                        earlier_path, later_path, label_path, root_path=dataset.root_path
                    )
                    
                    # Run each pipeline
                    for pipeline_name in ['Baseline', 'CVA', 'IR-MAD']:
                        try:
                            pipeline = get_pipeline(pipeline_name)
                            predicted_mask, _ = pipeline.run(earlier_img, later_img)
                            metrics = calculate_metrics(predicted_mask, label_img)
                            
                            results[pipeline_name]['precision'].append(metrics.precision)
                            results[pipeline_name]['recall'].append(metrics.recall)
                            results[pipeline_name]['f1'].append(metrics.f1)
                            results[pipeline_name]['iou'].append(metrics.iou)
                        except Exception as e:
                            st.warning(f"Error running {pipeline_name} on {pair_names[idx]}: {str(e)}")
                
                # Calculate averages
                avg_results = {}
                for pipeline_name in results:
                    avg_results[pipeline_name] = {
                        'precision': np.mean(results[pipeline_name]['precision']) if results[pipeline_name]['precision'] else 0,
                        'recall': np.mean(results[pipeline_name]['recall']) if results[pipeline_name]['recall'] else 0,
                        'f1': np.mean(results[pipeline_name]['f1']) if results[pipeline_name]['f1'] else 0,
                        'iou': np.mean(results[pipeline_name]['iou']) if results[pipeline_name]['iou'] else 0,
                    }
                
                # Display results
                st.subheader("Average Metrics")
                
                # Create metrics table
                metrics_df = pd.DataFrame(avg_results).T
                st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['f1', 'iou']))
                
                # Plot comparison chart
                st.subheader("Metrics Comparison")
                fig = plot_metrics(avg_results, metric_names=['f1', 'iou'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed metrics table
                st.subheader("Detailed Metrics")
                detailed_df = pd.DataFrame({
                    (pipeline, metric): results[pipeline][metric]
                    for pipeline in results
                    for metric in ['precision', 'recall', 'f1', 'iou']
                })
                detailed_df.index = [pair_names[i] for i in selected_indices]
                st.dataframe(detailed_df)
                
                # Store results for export
                st.session_state.comparison_results = {
                    'average': avg_results,
                    'detailed': detailed_df.to_dict(),
                    'dataset': dataset_name,
                    'split': selected_split,
                    'images': [pair_names[i] for i in selected_indices]
                }
                
                st.success("Comparison completed!")
    
    except Exception as e:
        st.error(f"Error running comparison: {str(e)}")
        import traceback
        st.text(traceback.format_exc())

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
**ChangeMap IR-MAD** is a classical computer vision project for change detection in satellite imagery using:
- **Baseline**: Image differencing + Otsu + Morphology
- **CVA**: Change Vector Analysis
- **IR-MAD**: Iteratively Reweighted MAD

Built for final-year B.Tech CS project.
""")
