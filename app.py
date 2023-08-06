__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import streamlit.components.v1 as components
from streamlit_extras.switch_page_button import switch_page


st.set_page_config(page_title="여행스타그램",initial_sidebar_state="collapsed",layout="wide")
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

    
    

CDN_PATH = 'https://cdn.knightlab.com/libs/timeline3/latest'
CSS_PATH = 'timeline3/css/timeline.css'
JS_PATH = 'timeline3/js/timeline.js'

SOURCE_TYPE = 'json' # json or gdocs
JSON_PATH = 'timeline_nlp.json' # example json

TL_HEIGHT = 800 # px


# load data
json_text = ''
if SOURCE_TYPE == 'json':
    with open(JSON_PATH, "r") as f:
        json_text = f.read()
        source_param = 'timeline_json'
        source_block = f'var {source_param} = {json_text};'



# load css + js
css_block = f'<link title="timeline-styles" rel="stylesheet" href="{CDN_PATH}/css/timeline.css">'
js_block  = f'<script src="{CDN_PATH}/js/timeline.js"></script>'


# write html block
htmlcode = css_block + ''' 
''' + js_block + '''

    <div id='timeline-embed' style="width: 100%; height: '''+str(TL_HEIGHT)+'''px; margin: 1px;"></div>

    <script type="text/javascript">
        var additionalOptions = {
            start_at_end: false, is_embed:true,
        }
        '''+source_block+'''
        timeline = new TL.Timeline('timeline-embed', '''+source_param+''', additionalOptions);
    </script>'''


#UI sections
data = 'Data'
code = 'HTML Code'
line = 'Visualization'
about = 'About'

components.html(htmlcode, height=TL_HEIGHT,)


m = st.markdown("""

<style>
div.stButton > button:first-child {

    position: relative;
    display: inline-block;
    font-size: px;
    color: white;

    border-radius: 6px;
    transition: top .01s linear;
    text-shadow: 0 1px 0 rgba(0,0,0,0.15);
    background-color: #82D1E3;
    border: none;
}
</style>

  """, unsafe_allow_html=True)
col1, col2, col3, col4, col5 = st.columns([5,4,3,2,0.1])
with col5:
    want_to_contribute = st.button("▶")
    if want_to_contribute:
        switch_page("page2")

        

    
