import streamlit as st
from streamlit_elements import elements, sync, event
from types import SimpleNamespace
from streamlit_elements import mui
from uuid import uuid4
from abc import ABC, abstractmethod
from streamlit_elements import dashboard, mui
from contextlib import contextmanager
from langchain import LLMChain
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from streamlit_extras.switch_page_button import switch_page
st.set_page_config(page_title="여행스타그램",initial_sidebar_state="collapsed",layout="wide")

class Dashboard:
    DRAGGABLE_CLASS = "draggable"

    def __init__(self):
        self._layout = []

    def _register(self, item):
        self._layout.append(item)

    @contextmanager
    def __call__(self, **props):
        # Draggable classname query selector.
        props["draggableHandle"] = f".{Dashboard.DRAGGABLE_CLASS}"

        with dashboard.Grid(self._layout, **props):
            yield

    class Item(ABC):

        def __init__(self, board, x, y, w, h,title,hashtag,img, **item_props):
            self._key = str(uuid4())
            self._draggable_class = Dashboard.DRAGGABLE_CLASS
            self._dark_mode = True
            
            ## --
            self.s_title = title
            self.hashtag = hashtag
            self.img = img
            ## --
            
            board._register(dashboard.Item(self._key, x, y, w, h, **item_props))

        def _switch_theme(self):
            self._dark_mode = not self._dark_mode

        @contextmanager
        def title_bar(self, padding="5px 15px 5px 15px", dark_switcher=True):
            with mui.Stack(
                className=self._draggable_class,
                alignItems="center",
                direction="row",
                spacing=1,
                sx={
                    "padding": padding,
                    "borderBottom": 1,
                    "borderColor": "divider",
                },
            ):
                yield

                if dark_switcher:
                    if self._dark_mode:
                        mui.IconButton(mui.icon.DarkMode, onClick=self._switch_theme)
                    else:
                        mui.IconButton(mui.icon.LightMode, sx={"color": "#ffc107"}, onClick=self._switch_theme)

        @abstractmethod
        def __call__(self):
            """Show elements."""
            raise NotImplementedError


class Card(Dashboard.Item):

    DEFAULT_CONTENT = (
        "This impressive paella is a perfect party dish and a fun meal to cook "
        "together with your guests. Add 1 cup of frozen peas along with the mussels, "
        "if you like."
    )

    def __call__(self, content):
        with mui.Card(key=self._key, sx={"display": "flex", "flexDirection": "column", "borderRadius": 3, "overflow": "hidden"}, elevation=1):
            mui.CardHeader(
                title=self.s_title,
                subheader=self.hashtag,
                avatar=mui.Avatar("S", sx={"bgcolor": "#82D1E3"}),
                action=mui.IconButton(mui.icon.MoreVert),
                className=self._draggable_class,
            )
            mui.CardMedia(
                component="img",
                height=194,
                image=self.img,
                alt="조회 이미지",
            )

            with mui.CardContent(sx={"flex": 1}):
                mui.Typography(content)

            with mui.CardActions(disableSpacing=True):
                mui.IconButton(mui.icon.Favorite)
                mui.IconButton(mui.icon.Share)

def instagram_gpt(text):
    instagram_template = """다음 내용을 220자 이내의 인스타그램 피드처럼 바꿔주세요. {text}"""
    instagram_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=PromptTemplate.from_template(instagram_template))
    return instagram_chain({'text' : text})['text']


st.markdown("""
    <style>
        [data-testid="collapsedControl"] {
            display: none
        }
    </style>
""",
    unsafe_allow_html=True,
)
st.markdown("""

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

empyt1,con,empty2 = st.columns([30,20,30])
e1,first,e2 = st.columns([160,20,160])



total_number = 0
keys_name = list(st.session_state.ans.keys())
total_context = []
total_hashtag= []
total_name = []
total_imgs = []
total_answer = []
set_chagne=False

if set_chagne:
    with first:
        want_to_contribute = st.button("다시하기")
        if want_to_contribute:
            switch_page("page2")



for i in keys_name:
    total_number += len(st.session_state.data[i])

for i in keys_name:
    if total_number == 1:
        total_answer += [st.session_state.ans[i].split(f'조회된 개수보다 추천 수가 많아서 조회된 개수 {total_number}개 내에서 추천하는 것으로 변경되었습니다. \n\n ')[-1]]
    else:
        total_answer += st.session_state.ans[i].split('\n\n')

for i in keys_name:
    for j in st.session_state.data[i]:
        total_name.append(j.page_content)
        total_imgs.append(j.metadata['img'])
        total_hashtag.append(f'#{i} #{j.page_content} #여행스타그램')

# print(total_name)

if len(total_answer) != 1:
    for i in total_name:
        for j in total_answer:
            if i in j:
                total_context.append(instagram_gpt(j))
                break
else:
    total_context.append(instagram_gpt(total_answer[0]))


# st.write(st.session_state.ans)
# st.write(st.session_state.data)

# st.write(total_answer)
# st.write(total_context)



    

if  total_number != 0:
    with con:
        with elements("style_mui_sx"):
            mui.Box(
                f"#{st.session_state.hashtag}  #여행스타그램",
                sx={
                    "fontWeight":'bold',
                    "textAlign": "center",
                    "bgcolor": "#ededed",
                    "boxShadow": 1,
                    "fontSize" : 20,
                    "borderRadius": 2,
                    'alignItems': 'center' ,
                    "p": 2,
                    # "minWidth": 80,
                    # "width" : 1,
                    # "justifyContent":"center"
                }
            )
if "w" not in st.session_state:
    # title,hashtag,img
    board = Dashboard()
    new_dic = {'dashboard' : board}
    cnt = 0
    for i in range(total_number):
        if cnt < 4:
            if cnt == 0:
                new_dic[f'card{i}'] =  Card(board, 0, 0, 3, 10, total_name[i], total_hashtag[i], total_imgs[i],minW=2, minH=4)
            if cnt == 1:
                new_dic[f'card{i}'] =  Card(board, 3, 0, 3, 10, total_name[i], total_hashtag[i], total_imgs[i],minW=2, minH=4)
            if cnt == 2:
                new_dic[f'card{i}'] =  Card(board, 6, 0, 3, 10, total_name[i], total_hashtag[i], total_imgs[i],minW=2, minH=4)
            if cnt == 3:
                new_dic[f'card{i}'] =  Card(board, 9, 0, 3, 10, total_name[i], total_hashtag[i], total_imgs[i],minW=2, minH=4)
            cnt += 1
        else:
            cnt =0
            new_dic[f'card{i}'] =  Card(board, 0, 0, 3, 10, total_name[i], total_hashtag[i], total_imgs[i],minW=2, minH=4)
            cnt += 1
    w = SimpleNamespace(**new_dic)
    st.session_state.w = w
else:
    w = st.session_state.w


if  total_number != 0:
    with elements("demo"):
        event.Hotkey("ctrl+s", sync(), bindInputs=True, overrideDefault=True)
        with w.dashboard(rowHeight=57):
            for i in range(len(new_dic)-1):
                text = total_context[i]
                text = text.replace('"',' ').strip()
                text = text.replace("'",' ').strip()
                eval(f'w.card{i}("""{text}""")')
            set_chagne = True




