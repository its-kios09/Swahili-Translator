import streamlit as st
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# load the trained model and tokenizer
model_path = "./models"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# create the translation pipeline
translator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
)

# define the translation function
def translate_text(text):
    translated_text = translator(text, max_length=128)[0]['generated_text']
    return translated_text

# create the Streamlit app
def main():
    st.set_page_config(page_title="English to Swahili Translation", page_icon="🌍", layout="wide")
    st.write("<h1 style='text-align: center;'>Welcome to Kenya 🇰🇪 🤝🏾</h1>", unsafe_allow_html=True)
    st.write("<p style='text-align: center;'>🤖 Your Virtual Swahili Translator</p>", unsafe_allow_html=True)

    # allow user to input a sentence
    text = st.text_input("Enter text to translate", "")

    # generate and display the translation
    if st.button("Translate"):
        if text:
            with st.spinner("Translating..."):
                translated_text = translate_text(text)
            st.markdown("<h3 style='text-align: center;'>Translation:</h3>", unsafe_allow_html=True)
            df = pd.DataFrame({'Input Text': [text], 'Predicted Text': [translated_text]})
            styled_table = df.style \
                .set_table_styles([{                
                    'selector': 'th',                
                    'props': [                    
                        ('text-align', 'center'),                    
                        ('background-color', 'lightblue'),                    
                        ('color', 'white')
                    ]
                }, {
                    'selector': 'td',
                    'props': [
                        ('text-align', 'center')
                    ]
                }]) \
                .set_properties(**{'text-align': 'center', 'border-collapse': 'collapse', 'border': '1px solid black'}) \
                .set_table_attributes('border="1" class="dataframe" style="border-collapse: collapse; text-align: center; margin: auto; margin-top: 40px;"')
            styled_table = styled_table.set_table_styles([
                {'selector': 'th', 'props': [('text-align', 'center'), ('background-color', 'lightcoral'), ('color', 'white'), ('border', '1px solid white')]},
                {'selector': 'td', 'props': [('text-align', 'center'), ('border', '1px solid white')]}
            ])
            styled_table = styled_table.hide_index().set_properties(**{'border-collapse': 'collapse'})
            st.write("<div style='text-align: center;'>" + styled_table.render() + "</div>", unsafe_allow_html=True)
        else:
            st.warning("Please enter an English sentence to translate.")



if __name__ == "__main__":
    main()
