from features import load_gpt2, extract_features

model, tok = load_gpt2("gpt2")  # или твой ru-модельнейм
text = "Уважаемый клиент, для подтверждения операции пройдите по ссылке."

vec = extract_features(text, model, tok)
print("shape:", vec.shape)
print("features:", vec)