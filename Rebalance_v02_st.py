

import rebalancing_functions_st as reb
import streamlit as st

# Set page title
st.title('Portfolio Rebalancing based on Maximum Diversification Portfolio Optimization')
st.markdown('## ')

# Set page name and icon
st.set_page_config(
    page_title = 'Maximum Diversification',
    page_icon = 'j_fb.png',
)

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

# Centering logo in the sidebar
col1, col2, col3 = st.sidebar.beta_columns([3,7,1])
with col1:
    st.write("")
with col2:
    st.image('j.png', width = 90)
with col3:
    st.write("")
