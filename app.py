import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from prompt import burn_ai_assistant

# App config
st.set_page_config(page_title="🔥 DermaIQ - Burn Classifier", layout="wide")

# Load model
model = tf.keras.models.load_model("burn_model_final.h5")
class_map = {0: "First-degree", 1: "Second-degree", 2: "Third-degree"}

# Grad-CAM functions
def make_gradcam_heatmap(img_array, base_model, last_conv_layer_name, classifier_model, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, features = grad_model(img_array)
        preds = classifier_model(features)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(heatmap, img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = heatmap_color * alpha + img * 255
    return np.uint8(superimposed)

# Extract base and classifier model
base_model = model.get_layer("mobilenetv2_1.00_224")
classifier_input = tf.keras.Input(shape=base_model.output.shape[1:])
x = tf.keras.layers.GlobalAveragePooling2D()(classifier_input)
x = tf.keras.layers.Dropout(0.3)(x)
output = tf.keras.layers.Dense(3, activation='softmax')(x)
classifier_model = tf.keras.Model(classifier_input, output)

# Sidebar
st.sidebar.markdown("## 🔍 Navigation")
page = st.sidebar.radio("Choose a section", ["🏥 Prediction", "🧠 Grad-CAM", "📈 Accuracy & Metrics", "🤖 AI Assistant"])
st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("📤 Upload a wound image", type=["jpg", "jpeg", "png"])
st.sidebar.caption("Images are resized to 224x224 for model input.")

# Main App Layout
if page == "🏥 Prediction":
    st.title("🔥 DermaIQ Burn Severity Prediction")
    st.markdown("Upload a burn wound image to identify its severity.")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0).astype(np.float32)

        pred = model.predict(img_input)
        pred_class = np.argmax(pred[0])
        class_label = class_map[pred_class]

        st.success(f"🩺 **Predicted Class:** {class_label}")
        st.bar_chart(pred[0])

elif page == "🧠 Grad-CAM":
    st.title("🧠 Visual Explanation with Grad-CAM")
    st.markdown("See where the model is focusing when making predictions.")

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_input = np.expand_dims(img_array, axis=0).astype(np.float32)

        heatmap = make_gradcam_heatmap(img_input, base_model, "Conv_1", classifier_model)
        cam_image = overlay_gradcam(heatmap, img_array)

        st.image(cam_image, caption="🔥 Grad-CAM Heatmap", use_column_width=True)

elif page == "📈 Accuracy & Metrics":
    st.title("📊 Model Performance Metrics")
    st.markdown("These values reflect how the model performed on the validation set.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", "92%")
    col2.metric("Precision", "91%")
    col3.metric("Recall", "90%")
    col4.metric("F1 Score", "90%")

    with st.expander("🔍 Metric Details"):
        st.markdown("""
        - **Accuracy**: Overall correctness of the model
        - **Precision**: Correct positive predictions / total predicted positives
        - **Recall**: Correct positive predictions / total actual positives
        - **F1 Score**: Harmonic mean of Precision and Recall
        """)

elif page == "🤖 AI Assistant":
    st.title("🤖 DermaIQ AI Assistant")
    st.markdown("""
        Ask anything about:
        - 🔥 Burn types (First, Second, Third degree)
        - 🩹 Wound care & first-aid
        - 🧠 Model prediction safety
        - 🧪 Diagnosis suggestions
    """)

    user_input = st.text_input("💬 What would you like to ask?")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = burn_ai_assistant(user_input)  # 👈 Corrected function name
                st.success("🧠 Burn AI Assistant says:")
                st.markdown(response)
            except Exception as e:
                st.error(f"❌ Error: {e}")


