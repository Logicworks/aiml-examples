import streamlit as st
import time

# Import the ConfluenceQA class
from bedrock_confluence_qa import BedrockConfluenceQA

st.set_page_config(
    page_title='AWS Bedrock powered Q&A Bot for Confluence Spaces',
    page_icon='🤖',
    layout='wide',
    initial_sidebar_state='auto',
)
if "config" not in st.session_state:
    st.session_state["config"] = {}
if "confluence_qa" not in st.session_state:
    st.session_state["confluence_qa"] = None
st.session_state['model'] = 'amazon.titan-tg1-large'
@st.cache_resource
def load_confluence(config):
    st.write("loading the confluence page")
    confluence_qa = BedrockConfluenceQA(config=config)
    confluence_qa.init_embeddings()
    st.session_state['parameters'] = {
        "model": 'amazon.titan-tg1-large',
        "max_token_count": 4096,
        "top_p": 1,
        "top_k": 1,
    }
    confluence_qa.init_models(st.session_state['parameters'])
    confluence_qa.vector_db_confluence_docs()
    confluence_qa.retreival_qa_chain()
    st.write(f"Model {confluence_qa.model_id} used")
    return confluence_qa

with st.sidebar.form(key ='Form1'):
    st.markdown('## Add your configs')
    confluence_url = st.text_input("paste the confluence URL", "https://xxx.atlassian.net/wiki")
    username = st.text_input(label="confluence username",
                             help="leave blank if confluence page is public",
                             type="password")
    space_key = st.text_input(label="confluence space",
                             help="Space of Confluence")
    api_key = st.text_input(label="confluence api key",
                            help="leave blank if confluence page is public",
                            type="password")
    submitted1 = st.form_submit_button(label='Submit')

    if submitted1 and confluence_url and space_key:
        st.session_state["config"] = {
            "persist_directory": None,
            "confluence_url": confluence_url,
            "username": username if username != "" else None,
            "api_key": api_key if api_key != "" else None,
            "space_key": space_key,
            "model_id": st.session_state['model'],
        }
        with st.spinner(text="Ingesting Confluence..."):
            st.session_state["confluence_qa"]  = load_confluence(st.session_state["config"])
        st.write("Confluence Space Ingested")
        
with st.sidebar.form(key ='Form3'):
    st.markdown('## Inference configuration')
    model = st.selectbox(
        'Select AWS foundation model',
        ('amazon.titan-tg1-large',
         'anthropic.claude-v2',
         'anthropic.claude-instant-v1',
         'anthropic.claude-v1',
         'ai21.j2-ultra',
         'ai21.j2-mid',
        ))
    max_token_count = st.number_input("Maximum length", min_value=0, max_value=9999, value=4096, help="leave blank if confluence page is public")
    temperature = st.number_input(label="Temprature", step=0.1, min_value=0.1, max_value=1.0, value=1.0, help="The Temperature value ranges from 0 to 1, with 0 being the most deterministic and 1 being the most creative.")
    top_p = st.number_input(label="TopP", value=1.0,  step=0.1, min_value=0.1, max_value=1.0, help="")
    top_k = st.number_input(label="TopK", value=250,  step=1, min_value=0, max_value=500, help="Only used by claude")
    
    
    
    submitted3 = st.form_submit_button(label='Submit')
    if submitted3 and max_token_count > 0 and top_p > 0 and temperature > 0:
        st.session_state['parameters'] = {
            "model": model,
            "max_token_count": max_token_count,
            "top_p": top_p,
            "top_k": top_k,
        }
        st.session_state["confluence_qa"].init_models(st.session_state['parameters'])
        st.session_state["confluence_qa"].retreival_qa_chain()
        st.write('You selected:', model)

    
st.title("AWS Bedrock powered Q&A Bot for Confluence Spaces")

question = st.text_input('Ask a question', "What is the focus of tasks during the first week of onboarding?")

if st.button('Get Answer', key='button2'):
    with st.spinner(text="Asking LLM..."):
        confluence_qa = st.session_state.get("confluence_qa")
        if confluence_qa is not None:
            with st.chat_message("assistant"):
                start_time = time.time()
                result = confluence_qa.answer_confluence(question)
                model_id = confluence_qa.model_id
                end_time = time.time()
                execution_time = round(end_time - start_time, 2)
                st.markdown(result['result'])
            with st.expander("#### Sources"):
                st.markdown(f"Took {execution_time} seconds using {model_id}")
                for idx, source in enumerate(result['source_documents'][:3]):
                    st.markdown(f"**{idx + 1}:** {source.page_content}")
                    st.markdown(f"{source.metadata['source']}")
                    st.markdown("---")
        else:
            st.write("Please load Confluence page first.")