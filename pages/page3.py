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
from streamlit_js_eval import streamlit_js_eval
import  streamlit_vertical_slider  as svs
import streamlit_ext as ste
import streamlit.components.v1 as components
st.set_page_config(page_title="여행스타그램",initial_sidebar_state="collapsed",layout="wide")

if 'page' not in st.session_state:
    st.session_state.page = 0
if "back_page" not in st.session_state:
    st.session_state.gpt = {}
    st.session_state.back_page = 1
    st.session_state.go_back_page = False
    st.session_state.go_next_page = False
    
    
    
if st.session_state.go_back_page == True:
    st.session_state.go_back_page = False
    st.session_state.three_to_second = 1
    switch_page("page2")

if st.session_state.go_next_page == True:
    st.session_state.go_next_page = False
    switch_page("page4")
    
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

# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

def instagram_gpt(text):
    instagram_template = """다음 내용을 220자 이내의 인스타그램 피드처럼 바꿔주세요. {text}"""
    instagram_chain = LLMChain(llm=ChatOpenAI(temperature=0), prompt=PromptTemplate.from_template(instagram_template))
    return instagram_chain({'text' : text})['text']

def swich_to_next():
    st.session_state.go_back_page = True
    streamlit_js_eval(js_expressions="parent.window.location.reload()")    

def add_choice():
    st.session_state.go_next_page = True
    for i in range(len(total_name)):
        st.session_state.data[f'set{i}'] += int(eval(f'st.session_state.select{i}').split('점')[0])
    st.session_state.data['road'] += st.session_state.road
    streamlit_js_eval(js_expressions="parent.window.location.reload()")



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





# nex11, next15, next16 = st.columns([160,20,160])
# with next15:
#     st.write('여행스타그램 생성중...')
    
# loading_placeholder = st.empty()

# Some code

def trip_instagram():    
    # with st.form('여행스타그램 form', clear_on_submit=True):
    empyt1,con,empty2 = st.columns([30,20,30])
    e1,first,e2 = st.columns([160,20,160])

    

    total_number = 0
    keys_name = list(st.session_state.ans.keys())
    total_context = []
    total_hashtag= []
    total_name = []
    total_imgs = []
    total_answer = []
    


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
                    st.session_state.gpt[f'{i}'] = [total_context[-1]]
                    break
    else:
        total_context.append(instagram_gpt(total_answer[0]))
        st.session_state.gpt[i] = [total_context[-1]]


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
                        # "fontWeight":'bold',
                        "textAlign": "center",
                        "bgcolor": "#94d0e0",
                        "boxShadow": 1,
                        "fontSize" : 20,
                        "borderRadius": 2,
                        'alignItems': 'center' ,
                        "p": 2,
                        "color": 'white'
                        # "minWidth": 80,
                        # "width" : 1,
                        # "justifyContent":"center"
                    }
                )
            # with first:
            #     next_page_btn = st.form_submit_button("다시하기", on_click=swich_to_next)
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
                    
    return total_name

def trip_select_number(total_name):
    with st.form('여행 경로 form'):
        
        option_list = [f"{i}점" for i in range(1,11)]
        st.session_state.next_data = {}
        st.markdown('### 여행 경로 추천을 위한 선호도를 입력해 주세요. (1~10점 중복가능)')

        
        st.write(' ')
        st.write(' ')
        st.session_state.next_data['trip_name'] = [i for i in total_name]
        # st.session_state.next_data['trip_num'] = []
        st.session_state.data = {}
        for i in range(len(total_name)):
            st.session_state.data[f'set{i}'] = 0
        st.session_state.data['road'] = 0
        min_value = 1
        max_value = 10
       
            
        

        if len(total_name) > 4:
            col_num = (len(total_name)//4)  + (len(total_name)%4) 
            
            for i in range(col_num):
                globals()[f'columns{i}'] = st.columns(4)
            # columns1 =  st.columns(4)
            # columns2 = st.columns(len(total_name) - 4)
            
            cnt = 0
            ncol = 0
            for idx, name in enumerate(total_name):
                if cnt >= 4:
                    if cnt %4 == 0:
                        ncol += 1
                    with globals()[f'columns{ncol}'][idx - (4*ncol)]:
                        #st.session_state.next_data['trip_num'].append(st.number_input(label = '', min_value=1, max_value=10,label_visibility="collapsed",value=3, key=idx))
                        #globals()[f'slid_trip{idx}'] = ste.slider(f':green[{name}]', 1, 10, 5, key=f'{idx}')
                        st.selectbox(f':green[{name}]', option_list, key=f'select{idx}')
                        #globals()[f'slid_trip{idx}'] = svs.vertical_slider(key=f'set_{idx}', default_value=5, step=1, min_value=min_value, max_value=max_value)
                else:
                    with globals()[f'columns{ncol}'][idx]:
                        #st.session_state.next_data['trip_num'].append(st.number_input(label = '', min_value=1, max_value=10,label_visibility="collapsed",value=3, key=idx))
                        #globals()[f'slid_trip{idx}'] = ste.slider(f':green[{name}]', 1, 10, 5, key=f'{idx}')
                        st.selectbox(f':green[{name}]', option_list, key=f'select{idx}')
                        #globals()[f'slid_trip{idx}'] = svs.vertical_slider(key=f'set_{idx}', default_value=5, step=1, min_value=min_value, max_value=max_value)               
                cnt +=1
        else:
            columns =  st.columns(len(total_name))
            for idx, name in enumerate(total_name):
                with columns[idx]:
                    #st.session_state.next_data['trip_num'].append(st.number_input(label = '', min_value=1, max_value=10,label_visibility="collapsed",value=3, key=idx))
                    #globals()[f'slid_trip{idx}'] = ste.slider(f':green[{name}]', 1, 10, 5, key=f'{idx}')
                    st.selectbox(f':green[{name}]', option_list, key=f'select{idx}')
            #               globals()[f'slid_trip{idx}'] = svs.vertical_slider(key=f'set_{idx}', default_value=5, step=1, min_value=min_value, max_value=max_value)
        st.write(' ')
        st.write(' ')
        st.write(' ')
        columns2 = st.columns(1)
        columns3 = st.columns(3)

        with columns2[0]:
            st.divider()
        with columns3[1]:
            st.number_input(':green[여행 가능한 최대 거리를 입력해주세요. (km단위)]',min_value=5, max_value=100, step=1,key='road')
        seper1, seper2, seper3 = st.columns([200,40,200])
        with seper2:
            st.form_submit_button(label='경로기반 추천', on_click=add_choice)
            
            # print(st.session_state['pdf_data'])
            # print('-'*30)
            # print(st.session_state.gpt)
            
            
            
        # trip_numbers = []
        # for idx in range(len(total_name)):
        #     print(globals()[f'slid_trip{idx}'])
        #     if globals()[f'slid_trip{idx}'] == None:
        #         globals()[f'slid_trip{idx}'] = 5
        #     trip_numbers.append(globals()[f'slid_trip{idx}'])
            
        
        

        # st.write(f'{total_name.pop(0)}')
        # option0 = st.selectbox('해당 여행지의 선호도 점수 :', option_list, key=0)
        # for i in range(len(total_name)):
        #     st.write(f'{total_name[i]}')
        #   
        # st.session_state.go_next_page = True
        # st.session_state.go_next_page['trip_num'] = [int(globals()[f'option{i}'].split('점')[0])  for i in range(len(total_name) + 1)]
        #submit_btn = st.form_submit_button(label='여행 경로 추천', on_click=swich_to_next2)
        # print([globals()[f'slid_trip{i}']  for i in range(len(total_name))])
        # print(st.session_state.select0, st.session_state.select1)

        # submit_btn = st.form_submit_button(label='여행 경로 추천', on_click=add_choice)
                

    # if set_chagne:
        


# lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
# lottie_json = load_lottieurl(lottie_url)


# import streamlit as st



# "# Center an image when in fullscreen"
# "Images (and most elements in general) are always aligned to the left"
# st.image("https://placekitten.com/g/200/200")


cols_place = st.empty()
n1, n2, n3 = cols_place.columns([200,40,200])
placeholder = st.empty()
loading_place = st.empty()
set_chagne=False
loading_one = 0

with placeholder.container():
    with n2:
        instagram_make = st.button('여행스타그램 생성',key='mstar')


#If btn is pressed or True
if instagram_make:
    # n1, n2, n3 = st.columns([160,20,160])
    # loading_one += 1
    with loading_place.container():
            components.html(
                """
                <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script> 
                <lottie-player src="https://lottie.host/85573875-9faa-4c85-a54e-afbb969a83d6/763ZtSvSup.json" background="transparent" speed="1" style="width: 50%; height: 50%;  display: table; margin-left: auto; margin-right: auto; " loop autoplay></lottie-player>
                """,
                height=1000,
            )
        
    total_name = trip_instagram()
    trip_select_number(total_name)
    if total_name:
        cols_place.empty()
        placeholder.empty()
        loading_place.empty()
    
    
    

# instagram_make = st.button('여행스타그램 생성',key='mstar')
# if instagram_make:
#     n1, n2, n3 = st.columns([160,20,160])
#     with n2:
#         loading = st.image('https://media.giphy.com/media/3oEjI6SIIHBdRxXI40/giphy.gif')
#     trip_instagram()
#     set_chagne = True    
#     loading.empty()
#     instagram_make
        
# if trip_instagram():
#     set_chagne = True    
#     loading_placeholder.empty()



        # if submit_btn:
            # return [globals()[f'slid_trip{i}']  for i in range(len(total_name))]
        
        

# select_boxs()
# print(select_boxs())
# # if set_chagne:
#     # nex1, next2, next3 = st.columns([160,20,160])
# nex4, next5, next6 = st.columns([160,20,160])
# option_list = [f"{i}점" for i in range(1,11)]
# st.session_state.next_data = {}
# with st.form('여행 경로 form'):
#     st.markdown('### 여행 경로 추천을 위한 선호도를 입력해 주세요. (1~10점 중복가능)')

    
#     st.write(' ')
#     st.write(' ')
#     st.session_state.next_data['trip_name'] = [i for i in total_name]
#     st.session_state.next_data['trip_num'] = []
#     min_value = 1
#     max_value = 10

#     if len(total_name) > 4:
#         col_num = (len(total_name)//4)  + (len(total_name)%4) 
        
#         for i in range(col_num):
#             globals()[f'columns{i}'] = st.columns(4)
#         # columns1 =  st.columns(4)
#         # columns2 = st.columns(len(total_name) - 4)
        
#         cnt = 0
#         ncol = 0
#         for idx, name in enumerate(total_name):
#             if cnt >= 4:
#                 if cnt %4 == 0:
#                     ncol += 1
#                 with globals()[f'columns{ncol}'][idx - (4*ncol)]:
#                     #st.session_state.next_data['trip_num'].append(st.number_input(label = '', min_value=1, max_value=10,label_visibility="collapsed",value=3, key=idx))
#                     #globals()[f'slid_trip{idx}'] = ste.slider(f':green[{name}]', 1, 10, 5, key=f'{idx}')
#                     globals()[f'slid_trip{idx}'] =st.selectbox(f':green[{name}]', option_list, key=f'{idx}')
#                     #globals()[f'slid_trip{idx}'] = svs.vertical_slider(key=f'set_{idx}', default_value=5, step=1, min_value=min_value, max_value=max_value)
#             else:
#                 with globals()[f'columns{ncol}'][idx]:
#                     #st.session_state.next_data['trip_num'].append(st.number_input(label = '', min_value=1, max_value=10,label_visibility="collapsed",value=3, key=idx))
#                     #globals()[f'slid_trip{idx}'] = ste.slider(f':green[{name}]', 1, 10, 5, key=f'{idx}')
#                     globals()[f'slid_trip{idx}'] =st.selectbox(f':green[{name}]', option_list, key=f'{idx}')
#                     #globals()[f'slid_trip{idx}'] = svs.vertical_slider(key=f'set_{idx}', default_value=5, step=1, min_value=min_value, max_value=max_value)               
#             cnt +=1
#     else:
#         columns =  st.columns(len(total_name))
#         for idx, name in enumerate(total_name):
#             with columns[idx]:
#                 #st.session_state.next_data['trip_num'].append(st.number_input(label = '', min_value=1, max_value=10,label_visibility="collapsed",value=3, key=idx))
#                 #globals()[f'slid_trip{idx}'] = ste.slider(f':green[{name}]', 1, 10, 5, key=f'{idx}')
#                 globals()[f'slid_trip{idx}'] =st.selectbox(f':green[{name}]', option_list, key=f'{idx}')
#         #               globals()[f'slid_trip{idx}'] = svs.vertical_slider(key=f'set_{idx}', default_value=5, step=1, min_value=min_value, max_value=max_value)
    
#     # trip_numbers = []
#     # for idx in range(len(total_name)):
#     #     print(globals()[f'slid_trip{idx}'])
#     #     if globals()[f'slid_trip{idx}'] == None:
#     #         globals()[f'slid_trip{idx}'] = 5
#     #     trip_numbers.append(globals()[f'slid_trip{idx}'])
        
    
    

#     # st.write(f'{total_name.pop(0)}')
#     # option0 = st.selectbox('해당 여행지의 선호도 점수 :', option_list, key=0)
#     # for i in range(len(total_name)):
#     #     st.write(f'{total_name[i]}')
#     #   
#     # st.session_state.go_next_page = True
#     # st.session_state.go_next_page['trip_num'] = [int(globals()[f'option{i}'].split('점')[0])  for i in range(len(total_name) + 1)]
#     #submit_btn = st.form_submit_button(label='여행 경로 추천', on_click=swich_to_next2)
#     print([globals()[f'slid_trip{i}']  for i in range(len(total_name))])
#     submit_btn = st.form_submit_button(label='여행 경로 추천')
#     if submit_btn:
        
#         st.write([globals()[f'slid_trip{i}']  for i in range(len(total_name))])
    