import pandas as pd
import csv
import os
from finvizfinance.screener.overview import Overview
from transformers import pipeline
import yfinance as yf #download news on market data from the Yahoo! Finance

from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()
# Get an environment variable
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

template = """
You are a finance assistant that can analyse text and decide if according the text it is worth to purchase the stock or not.
Provide a short answer if it is worth to purchase the stock or not or it is not clear from the text.

Text: {text}
"""

# Set up the prompt and LLM chain
prompt = PromptTemplate(template=template, input_variables=["text"])
chain = LLMChain(prompt=prompt, llm=llm)

filters_dict = {'Debt/Equity':'Under 1', 
                    'PEG':'Low (<1)', 
                    'Operating Margin':'Positive (>0%)', 
                    'P/B':'Low (<1)',
                    'P/E':'Low (<15)',
                    'InsiderTransactions':'Positive (>0%)'}

def get_undervalued_stocks():
    """
    Returns a list of tickers with:
    
    - Positive Operating Margin
    - Debt-to-Equity ratio under 1
    - Low P/B (under 1)
    - Low P/E ratio (under 15)
    - Low PEG ratio (under 1)
    - Positive Insider Transactions
    """
    foverview = Overview()
    foverview.set_filter(filters_dict=filters_dict)
    df_overview = foverview.screener_view()
    tickers = df_overview.Ticker.to_list()
    return tickers

hot_stocks = get_undervalued_stocks()

pipe = pipeline("text-classification", model="ProsusAI/finbert")

def get_ticker_news_sentiment(ticker):
    """
    Returns a Pandas dataframe of the given ticker's most recent news article headlines,
    with the overal sentiment of each article.

    Args:
        ticker (string)

    Returns:
        pd.DataFrame: {'Date', 'Article title', Article sentiment'}
    """
    ticker_news = yf.Ticker(ticker)
    news_list = ticker_news.get_news()
    #extractor = Goose()
    results=[]
    for dic in news_list:
      title = dic['content']['title']
      summary = dic['content']['summary']
      text = title+' '+ summary
      label = pipe(text)[0]['label']
      response = chain.run(text=text)

      results.append({'YF Article':f'{text}',
                         'LLM generation':f'{response}',
                         'LLM sentiment':label})
    df = pd.DataFrame(results)
    return df
  
selected_custom_name = st.sidebar.selectbox('Ticker List', ['', *hot_stocks])
  
user_input = st.text_area("Enter Ticker")
button = st.button("Generate Response")
if user_input and button:
    summary = get_ticker_news_sentiment(user_input)
    st.write("Summary : ", summary)
  
# Main Streamlit app
def main():
    st.title("Stock Screener Using LLM")
    with st.sidebar:
        st.title('💬 Hot Stocks App')
        st.markdown('''
        ## About
        Enter Ticker to get results
        ''')
        st.write('Made by Noa Cohen')
       

if __name__ == "__main__":
    main()
