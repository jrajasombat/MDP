

import rebalancing_functions_st as reb
import streamlit as st
import os
import base64

# Set page name and icon
st.set_page_config(
    page_title = 'Maximum Diversification',
    page_icon = 'j_fb.png',
)

# Set page title
st.title('Portfolio Rebalancing based on Maximum Diversification Portfolio Optimization')
st.markdown('## ')

# Initiate sidebar
st.sidebar.markdown('## Select Parameters')

# Call the rebalancing functions
coi = ['Stocks','Bonds','Gold']
reb.rebalance(coi)

# Configure sidebar
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## Discussion')
st.sidebar.markdown('* One approach to reduce risk when construcing an investment portfolio \
is __maximum diversification portfolio optimization__. In particular, this strategy maximizes \
a diversification ratio and does not take into account expected returns.')
st.sidebar.markdown('* The __diversification ratio__ is defined as the ratio of the weighted average \
of all the volatilites in the portfolio divided by the total portfolio volatility.')
st.sidebar.markdown('* This web app optimizes a portfolio of stocks, bonds, and gold at regular \
intervals.')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## Appendix')
st.sidebar.image('div_ratio.png')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')
st.sidebar.markdown('## ')

# Logo in the sidebar
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

@st.cache(allow_output_mutation=True)
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code

png_html = get_img_with_href('j.png', 'https://www.jimisinith.com/about')

st.sidebar.markdown('A web app developed by [Jimisi Rajasombat](https://www.jimisinith.com)')
col1, col2, col3 = st.sidebar.beta_columns([3,7,1])
with col1:
    st.write("")
with col2:
    st.markdown(png_html, unsafe_allow_html=True)
with col3:
    st.write("")
