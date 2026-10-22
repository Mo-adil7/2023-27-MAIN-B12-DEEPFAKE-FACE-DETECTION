from keras_facenet import FaceNet
import numpy as np

print("Loading FaceNet model...")

embedder = FaceNet()

print("FaceNet model loaded successfully!")

# Create a dummy image to test the model interface.
dummy_face = np.zeros((160, 160, 3), dtype=np.uint8)

embeddings = embedder.embeddings([dummy_face])

print("Embedding shape:", embeddings.shape)
print("FaceNet test completed successfully!")