import wespeaker

model = wespeaker.load_model('english')
model.set_gpu(0)
# This will compute the similarity between the generated audio and the reference audio
similarity = model.compute_similarity('generated/small_metavoice.wav', 'reference.wav')
print(similarity)


import speechmetrics
window_length = 5 # seconds
metrics = speechmetrics.load('absolute', window_length)
scores = metrics("generated/small_metavoice.wav")
print(scores)
