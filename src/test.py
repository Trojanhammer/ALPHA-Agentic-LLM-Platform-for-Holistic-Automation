import google.generativeai as genai

genai.configure(api_key="AIzaSyAVonmSz9NoUWqwjryPS-PyaPtC8MBKejc")

for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
