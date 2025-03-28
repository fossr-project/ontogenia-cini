import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns

def generate_heatmap(similarity_matrix, title="Heatmap"):
    """
    Generate a heatmap using seaborn and return it as a base64-encoded PNG.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded

def save_heatmap_image(encoded_image, output_folder, name):
    """
    Save the base64 encoded image to a file.
    """
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    image_data = base64.b64decode(encoded_image)
    file_path = os.path.join(output_folder, name)
    with open(file_path, 'wb') as f:
        f.write(image_data)
    return file_path
