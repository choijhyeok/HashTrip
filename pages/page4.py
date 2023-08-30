import streamlit as st
import pandas as pd
import numpy as np
import random
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import random
import matplotlib.pyplot as plt
import numpy as np
from haversine import haversine
from itertools import product
import base64
from pathlib import Path
from glob import glob
import streamlit_ext as ste
import fitz
from weasyprint.text.fonts import FontConfiguration
from weasyprint import HTML, CSS
st.set_page_config(page_title="여행스타그램_page4",initial_sidebar_state="collapsed",layout="wide")



if "sec_number" not in st.session_state:
    st.session_state['sec_number'] = 0
    st.session_state['check_row'] = 0
    st.session_state['package_logs'] = 0
    
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }x
</style>
""",
    unsafe_allow_html=True,
)

button_css = st.markdown("""
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
col1, col2 = st.columns([50,50])




def make_html(html_string,data,gpt):
    for key in data.keys():
        html_string += f"<h3 id='{key}-1'><span>{key}</span></h3><p>"
        for idx, n in enumerate(data[key]['name']):
            if (idx !=0) and (idx%4 ==0) and (len(data[key]['name'])-1 != idx) :
                html_string += '</p><p>'
            html_string += f"<a href='{data[key]['blog'][idx]}'><span>{idx+1}.{n}</span></a><span>  </span>" 
        html_string += '</p><p>'

        for idx,img in enumerate(data[key]['img']):
            if (idx != 0) and (idx%4 ==0) and (len(data[key]['img'])-1 != idx):
                html_string += '</p><p>'
            html_string += f"<img id='img_size' src='{img}' width='200px'/>"
        html_string += '</p><p>&nbsp;</p>'
        gpt_ans = data[key]['gpt_ans'].replace('\n','<br>')
        html_string += "<blockquote><p><span>ChatGPT 추천이유</span></p><p><span>" + f"{gpt_ans}" +"</span></p></blockquote><p>&nbsp;</p>"


    # GPT -> instagram 변경 생성
    html_string += "<h2 id='chatgpt로-작성한-인스타그램-피드'><span>ChatGPT로 작성한 인스타그램 피드</span></h2>"

    for key,value in gpt.items():
        sep_value = value[0].replace('\n','<br>')
        html_string += f'<ul><li><span>{key}</span></li></ul>'
        html_string +="""
        <pre class="md-fences md-end-block ty-contain-cm modeLoaded" spellcheck="false"
                            lang=""><div class="CodeMirror cm-s-inner cm-s-null-scroll CodeMirror-wrap" lang=""><div style="overflow: hidden; position: relative; width: 3px; height: 0px; top: 9.52344px; left: 8px;"><textarea autocorrect="off" autocapitalize="off" spellcheck="false" tabindex="0" style="position: absolute; bottom: -1em; padding: 0px; width: 1000px; height: 1em; outline: none;"></textarea></div><div class="CodeMirror-scrollbar-filler" cm-not-content="true"></div><div class="CodeMirror-gutter-filler" cm-not-content="true"></div><div class="CodeMirror-scroll" tabindex="-1"><div class="CodeMirror-sizer" style="margin-left: 0px; margin-bottom: 0px; border-right-width: 0px; padding-right: 0px; padding-bottom: 0px;"><div style="position: relative; top: 0px;"><div class="CodeMirror-lines" role="presentation"><div role="presentation" style="position: relative; outline: none;"><div class="CodeMirror-measure"><pre><span>xxxxxxxxxx</span></pre>
                    </div>
                    <div class="CodeMirror-measure"></div>
                    <div style="position: relative; z-index: 1;"></div>
                    <div class="CodeMirror-code" role="presentation">
                        <div class="CodeMirror-activeline" style="position: relative;">
                            <div class="CodeMirror-activeline-background CodeMirror-linebackground"></div>
                            <div class="CodeMirror-gutter-background CodeMirror-activeline-gutter" style="left: 0px; width: 0px;">
                            </div>
        """
        html_string +=f'<pre class=" CodeMirror-line "role="presentation"><span role="presentation" style="padding-right: 0.1px;">{sep_value}</span></pre>'
        html_string +='''
                    </div>
                    </div>
                    </div>
                    </div>
                    </div>
                    </div>
                    <div style="position: absolute; height: 0px; width: 1px; border-bottom: 0px solid transparent; top: 23px;"></div>
                    <div class="CodeMirror-gutters" style="display: none; height: 23px;"></div>
                    </div>
                    </div>
                    </pre>
        '''
    html_string += '<p>&nbsp;</p>'

    # 유전알고리즘
    html_string += "<h2 id='유전-알고리즘을-통한-최적의-여행-조합-추천'><span>유전 알고리즘을 통한 최적의 여행 조합 추천</span></h2>"
    html_string += '<p><img src="result.png" referrerpolicy="no-referrer" alt="유전알고리즘 그래프"></p><p>&nbsp;</p></div></div></body>'
    return html_string

#-------#

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded
def img_to_html(img_path):
    img_html = "<img src='data:image/png;base64,{}' class='img-fluid'>".format(
      img_to_bytes(img_path)
    )
    return img_html
class Knapsack01Problem:

    def __init__(self):

        # initialize instance variables:
        self.items = []
        self.maxKm = 10
        self.budget = 0
        self.load_dataset = True
        self.df = []

    def __len__(self):
        """
        :return: the total number of items defined in the problem
        """
        return len(self.items)

    def getItems(self,data):
        self.items = data


    def getValue_all(self, float_list):

        totalKm = totalValue = 0
        for i in range(len(float_list)):
            order_id, km, value = self.items[i]
            if totalKm + float(float_list[i] * km) <= self.maxKm:
                totalKm += float(float_list[i] * km)
                totalValue += int(value)
        return totalKm



    def printItems_all(self, float_list, check_row):
        package_logs =[]

        totalKm = totalValue = 0
        for i in range(len(float_list)):
            order_id, km, value = self.items[i]
            if totalKm + float(km) <= self.maxKm:
                totalKm += float(km)
                totalValue += int(value)
                check_row.append(order_id)
                package_logs.append("- Adding row :{}: "
                      "km = {}, "
                      "value = {}, "
                      "portion = {}, "
                      "accumulated weight = {}, "
                      "accumulated value = {}".format(int(order_id), km,  int(value), float_list[i], totalKm, totalValue))
        package_logs.append("Total KM = {}, Total value = {}".format(totalKm, int(totalValue)))
        return check_row, package_logs
    
    
plt.rcParams['figure.figsize'] = [15, 8]


def list_in_tuple(data):
    all_list = []
    for i in range(len(data)):
        all_list.append(tuple(data.iloc[i]))

    return all_list



def DEAP_float(data):
    check_row = []
    while(True):
        MAX_GENERATIONS = 300
        POPULATION_SIZE = 30
        P_CROSSOVER = 0.9
        P_MUTATION = 0.1
        HALL_OF_FAME_SIZE = 1
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        knapsack = Knapsack01Problem()
        knapsack.getItems(data)


        # data = ran_weight(data)
        knapsack.getItems(list_in_tuple(knapsack.items[knapsack.items.columns[[0,-2,-1]]]))
        def knapsackValue_all(individual):
            return knapsack.getValue_all(individual),  # return a tuple



        genetic_tool = base.Toolbox()
        genetic_tool.register("attr_float", random.random)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        genetic_tool.register("individualCreator", tools.initRepeat, creator.Individual, genetic_tool.attr_float, len(knapsack))
        genetic_tool.register("populationCreator", tools.initRepeat, list, genetic_tool.individualCreator)
        genetic_tool.register("evaluate", knapsackValue_all)
        genetic_tool.register("select", tools.selTournament, tournsize=3)
        genetic_tool.register("mate", tools.cxTwoPoint)
        genetic_tool.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.2)


        population = genetic_tool.populationCreator(n=POPULATION_SIZE)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        hof = tools.HallOfFame(HALL_OF_FAME_SIZE)
        population, logbook = algorithms.eaSimple(population, genetic_tool, cxpb=P_CROSSOVER, mutpb=P_MUTATION,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hof, verbose=False)

        best = hof.items[0]

        maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")
        # print()


        plt.plot(maxFitnessValues, label='maxFitnessValues')
        plt.plot(meanFitnessValues, label='meanFitnessValues')
        plt.xlabel('Generation',fontsize=20)
        plt.ylabel('Max / Average Fitness',fontsize=20)
        # plt.title('Max and Average fitness over Generations',fontsize=30, fontweight='bold')
        plt.grid(True)
        plt.legend(fontsize=20, loc = 'lower right')
        plt.savefig('result.png', dpi=100)


        print("-- Best Ever Individual = ", best)
        print("-- Best Ever Fitness = ", best.fitness.values[0])

        print("-- Knapsack Items --")
        check_row, package_logs = knapsack.printItems_all(best, check_row)
        # st.pyplot()
        # plt.show()
        # plt.cla()

        best_avg = 0.0
        best_gen = 0

        for i in logbook:
            if(i["max"] <= knapsack.maxKm):
                if(best_avg < i['avg']):
                    best_avg = i['avg']
                    best_gen = i['gen']

        print()
        if MAX_GENERATIONS == best_gen:
            print('\n 현재 수행결과가 가장좋은 결과입니다.')
            return check_row, package_logs
            break
        else:
            print('최고의 설정은 best avg : {}, best gen : {} 입니다.'.format(best_avg,best_gen))
            MAX_GENERATIONS = best_gen
            return check_row, package_logs
            break

def product_sep(data,keys):
    all_vars = []
    all_y = []
    all_x = []
    for  i in keys:
        all_vars.append(data[i]['name'])
        all_y.append(data[i]['y'])
        all_x.append(data[i]['x'])
    

    if len(keys) == 2:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1])))
        for idx,i in enumerate(product_list):
            km_float = haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], abs(km_float), random.randint(1,10)))
        return df_list
    
    elif len(keys) == 3:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]],km_float, random.randint(1,10)))
        return df_list
    
    elif len(keys) == 4:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]],km_float, random.randint(1,10)))
        return df_list

    elif len(keys) == 5:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]],km_float, random.randint(1,10)))
        return df_list
    
    elif len(keys) == 6:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4], all_vars[5]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4]), all_vars[5].index(i[5])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km')) + abs(haversine((float(all_y[4][i[4]]), float(all_x[4][i[4]])), (float(all_y[5][i[5]]), float(all_x[5][i[5]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]], all_vars[5][i[5]],km_float, random.randint(1,10)))
        return df_list
    
    elif len(keys) == 7:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4], all_vars[5], all_vars[6]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4]), all_vars[5].index(i[5]), all_vars[6].index(i[6])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km')) + abs(haversine((float(all_y[4][i[4]]), float(all_x[4][i[4]])), (float(all_y[5][i[5]]), float(all_x[5][i[5]])), unit = 'km')) + abs(haversine((float(all_y[5][i[5]]), float(all_x[5][i[5]])), (float(all_y[6][i[6]]), float(all_x[6][i[6]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]], all_vars[5][i[5]], all_vars[6][i[6]],km_float, random.randint(1,10)))
        return df_list
    
    elif len(keys) == 8:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4], all_vars[5], all_vars[6], all_vars[7]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4]), all_vars[5].index(i[5]), all_vars[6].index(i[6]) , all_vars[7].index(i[7])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km')) + abs(haversine((float(all_y[4][i[4]]), float(all_x[4][i[4]])), (float(all_y[5][i[5]]), float(all_x[5][i[5]])), unit = 'km')) + abs(haversine((float(all_y[5][i[5]]), float(all_x[5][i[5]])), (float(all_y[6][i[6]]), float(all_x[6][i[6]])), unit = 'km')) + abs(haversine((float(all_y[6][i[6]]), float(all_x[6][i[6]])), (float(all_y[7][i[7]]), float(all_x[7][i[7]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]], all_vars[5][i[5]], all_vars[6][i[6]] ,all_vars[7][i[7]] ,km_float, random.randint(1,10)))
        return df_list

# data
# print(st.session_state.pdf_data)
# print(st.session_state.gpt)

htmlf = open('template.html')
html_string = htmlf.read()
cssf = open('template.css')
css_string = cssf.read()

data_df = product_sep(st.session_state.pdf_data,list(st.session_state.pdf_data.keys()))
data_df = pd.DataFrame(data_df)
with col1:
    seper1, seper2, seper3 = st.columns([120,60,120])
    with seper2:
        st.button('최적의 여행조합 추천', disabled=True)
    if st.session_state['sec_number']  == 0:
        st.session_state['sec_number'] +=1
        check_row, package_logs = DEAP_float(data_df.sample(frac=1))
        st.session_state['check_row'] = check_row
        st.session_state['package_logs'] = package_logs
        
        st.image('result.png')
        st.divider()
        seper11, seper22, seper33 = st.columns([30,200,30])
        with seper22:
            st.write('##### 최종 추천 결과')
            st.write(f'총 합이 10 km를 넘지않고 선호도가 최대인 여행지 조합 추천')
            st.write(f'{package_logs[-1]}, choice rows = {list(map(int,check_row))}')
            st.dataframe(data_df.iloc[check_row][data_df.columns[1:]])
    else:
        st.image('result.png')
        st.divider()
        seper11, seper22, seper33 = st.columns([30,200,30])
        with seper22:
            st.write('##### 최종 추천 결과')
            st.write(f'총 합이 10 km를 넘지않고 선호도가 최대인 여행지 조합 추천')
            st.write(f"{st.session_state['package_logs'][-1]}, choice rows = {list(map(int,st.session_state['check_row']))}")
            st.dataframe(data_df.iloc[st.session_state['check_row']][data_df.columns[1:]])
with col2:
    string_html = make_html(html_string,st.session_state.pdf_data,st.session_state.gpt)
    font_config = FontConfiguration()
    html = HTML(string=string_html, base_url='.')
    css = CSS(string=css_string, font_config=font_config)
    html.write_pdf('template.pdf', stylesheets=[css], font_config=font_config)
    pdf_doc = fitz.open('template.pdf')
    for i in pdf_doc:
        pix = i.get_pixmap()
        pix.save("page-%i.png" % i.number)
        
    seper4, seper5, seper6 = st.columns([120,80,120])
    seper44, seper55, seper66 = st.columns([30,120,30])
    seper7, seper8, seper9 = st.columns([120,80,120])   
    with seper5:
        st.button('여행스타그램 전체 과정 정리', disabled=True)
    with seper55:
        st.write(' ')
        st.write(' ')
        st.write(' ')
        image_list = glob(f'page-*.png')
        image_list.sort()
        number = ste.select_slider(
            '**HashTrip PDF 미리보기**',
            options=image_list,
            key="select_slider",
        )
        st.image(number, width=600)
    
    with seper8:
        with open('template.pdf', 'rb') as f:
            st.download_button(':blue[⬇ Download HashTrip PDF]', f, file_name='HashTrip.pdf')


# data = {'관광지': {'x': [127.2814498029, 127.293270883, 127.3009728808], 'y': [35.2836156407, 35.2874348574, 35.2913428821], 'img': ['http://tong.visitkorea.or.kr/cms/resource/37/1970937_image2_1.jpg', 'http://tong.visitkorea.or.kr/cms/resource/32/1609432_image2_1.jpg', 'https://blog.kakaocdn.net/dn/qNAeB/btsoSnQE2Pn/iot58TS4SvE41YtdKzhAok/img.jpg'], 'name': ['서산사(곡성)', '곡성 단군전', '곡성 메타세쿼이아길'], 'gpt_ans': '1. 서산사(곡성): 서산사는 곡성에 위치한 아름다운 사찰로, 자연과 조화를 이루는 풍경과 평화로운 분위기가 힐링과 휴식을 제공합니다. 사찰 내부에는 다양한 종류의 불상과 불교 문화유산을 감상할 수 있으며, 조용한 곳에서 명상을 즐길 수도 있습니다.\n\n2. 곡성 단군전: 곡성 단군전은 단군 이야기와 고려시대 역사를 배울 수 있는 곳으로, 역사적인 분위기와 아름다운 정원이 휴식과 힐링을 위한 좋은 장소입니다. 단군전 주변에는 산책로와 휴식 공간이 마련되어 있어 자연과 함께 힐링을 즐길 수 있습니다.\n\n3. 곡성 메타세쿼이아길: 곡성 메타세쿼이아길은 아름다운 메타세쿼이아 숲을 따라 산책할 수 있는 길로, 자연 속에서의 휴식과 힐링을 제공합니다. 숲 속을 걷는 동안 신선한 공기와 푸르른 풍경을 감상하며 스트레스를 풀고 마음을 가라앉힐 수 있습니다.\n\n이렇게 해시 태그 #힐링 #휴식을 기반으로 3개의 추천을 드렸습니다.'}, '문화시설': {'x': [127.3082114814, 127.3082975874, 127.2974151402], 'y': [35.2776556637, 35.2775723415, 35.2804829367], 'img': ['http://tong.visitkorea.or.kr/cms/resource/30/2789630_image2_1.jpg', 'http://tong.visitkorea.or.kr/cms/resource/59/2663259_image2_1.jpg', 'http://tong.visitkorea.or.kr/cms/resource/51/1971051_image2_1.jpg'], 'name': ['섬진강천적곤충관 (섬진강기차마을생태학습관)', '한국초콜릿연구소 뮤지엄 곡성지점', '곡성문화원'], 'gpt_ans': '1. 섬진강천적곤충관 (섬진강기차마을생태학습관): 이곳은 자연과 생태계에 관심이 있는 분들에게 추천하는 장소입니다. 섬진강 기차마을에 위치한 이 생태학습관은 다양한 천적곤충들을 관찰하고 배울 수 있는 곳으로, 자연 속에서 힐링과 휴식을 즐길 수 있습니다.\n\n2. 한국초콜릿연구소 뮤지엄 곡성지점: 이곳은 초콜릿을 사랑하는 분들에게 추천하는 장소입니다. 곡성에 위치한 한국초콜릿연구소 뮤지엄은 초콜릿의 역사와 제조과정을 배울 수 있는 곳으로, 힐링과 휴식을 즐기면서 초콜릿에 대한 새로운 지식을 얻을 수 있습니다.\n\n3. 곡성문화원: 이곳은 문화와 예술을 즐기고 싶은 분들에게 추천하는 장소입니다. 곡성에 위치한 곡성문화원은 다양한 전시와 공연을 즐길 수 있는 문화 공간으로, 힐링과 휴식을 취하면서 예술과 문화에 대한 새로운 경험을 할 수 있습니다.'}}
# # gpt = {{'서산사(곡성)': ['#서산사 #곡성 #사찰 #자연과조화 #평화로운분위기 #힐링 #휴식 #불상 #불교문화유산 #명상\n\n곡성에 위치한 아름다운 #서산사는 자연과 조화를 이루며 평화로운 분위기를 선사합니다. 사찰 내부에는 다양한 종류의 불상과 불교 문화유산을 감상할 수 있어요. 또한 조용한 곳에서 명상을 즐길 수도 있어 힐링과 휴식을 제공합니다. 곡성을 방문한다면 꼭 서산사를 찾아보세요! 🌿🙏🏻✨'], '곡성 단군전': ['🌸 곡성 단군전 🌸\n역사와 자연이 어우러진 힐링의 장소✨\n곡성 단군전은 단군 이야기와 고려시대 역사를 배울 수 있는 곳이야. 역사적인 분위기와 아름다운 정원이 너를 위한 휴식처야. 주변에는 산책로와 휴식 공간도 마련돼 있어 자연과 함께 힐링을 즐길 수 있어. 여기서 멋진 사진도 찍어서 인스타에 올려봐! 📸💖 #곡성단군전 #역사와자연 #힐링장소 #산책로 #휴식공간 #인스타그램'], '곡성 메타세쿼이아길': ['"곡성 메타세쿼이아길, 자연 속 힐링의 여정✨ 아름다운 메타세쿼이아 숲을 따라 산책하며 신선한 공기와 푸르른 풍경을 만끽해보세요. 숲 속에서 스트레스를 풀고 마음을 가라앉힐 수 있는 최고의 휴식처입니다.🌳💆\u200d♀️ 자연과 함께하는 힐링 타임을 즐겨보세요. #곡성메타세쿼이아길 #산책로 #자연휴식 #힐링"'], '섬진강천적곤충관 (섬진강기차마을생태학습관)': ['섬진강천적곤충관은 섬진강기차마을에 위치한 생태학습관으로, 자연과 생태계에 관심이 있는 분들에게 추천합니다. 이곳에서는 다양한 천적곤충들을 관찰하고 배울 수 있어요. 자연 속에서 힐링과 휴식을 즐길 수 있는 멋진 장소입니다. #섬진강천적곤충관 #생태학습관 #자연과생태계 #힐링과휴식'], '한국초콜릿연구소 뮤지엄 곡성지점': ['🍫 한국초콜릿연구소 뮤지엄 곡성지점 🍫\n초콜릿을 사랑하는 분들에게 추천하는 장소! 곡성에 위치한 한국초콜릿연구소 뮤지엄은 초콜릿의 역사와 제조과정을 배울 수 있는 곳이에요. 여기서는 힐링과 휴식을 즐기면서 초콜릿에 대한 새로운 지식을 얻을 수 있답니다. 🌱 초콜릿을 만드는 과정을 직접 체험하고, 다양한 종류의 초콜릿을 맛보며 달콤한 시간을 보낼 수 있어요. 🍬 또한, 아름다운 정원과 카페에서 휴식을 취하며 초콜릿에 대한 깊은 이해를 할 수 있어요. 이곳에서 달콤한 여행을 즐겨보세요! 😊💕 #한국초콜릿연구소 #곡성지점 #초콜릿뮤지엄 #초콜릿여행 #힐링 #휴식 #달콤한시간 #초콜릿사랑 #초콜릿체험 #초콜릿맛보기 #카페 #여행'], '곡성문화원': ['곡성문화원은 문화와 예술을 사랑하는 분들을 위한 최고의 장소입니다. 이곳에서는 다양한 전시와 공연을 즐길 수 있어요. 곡성에 위치한 이 문화 공간은 힐링과 휴식을 취하면서 예술과 문화에 대한 새로운 경험을 할 수 있는 곳이에요. 여기서 멋진 작품들을 감상하고 예술의 세계에 빠져들어보세요. 곡성문화원은 당신을 환영합니다! #곡성문화원 #문화공간 #전시 #공연 #예술 #힐링 #휴식 #새로운경험']}}




# data = {'관광지': {'x': [127.2814498029, 127.293270883, 127.3009728808], 'y': [35.2836156407, 35.2874348574, 35.2913428821], 'img': ['http://tong.visitkorea.or.kr/cms/resource/37/1970937_image2_1.jpg', 'http://tong.visitkorea.or.kr/cms/resource/32/1609432_image2_1.jpg', 'https://blog.kakaocdn.net/dn/qNAeB/btsoSnQE2Pn/iot58TS4SvE41YtdKzhAok/img.jpg'], 'name': ['서산사(곡성)', '곡성 단군전', '곡성 메타세쿼이아길'], 'gpt_ans': '1. 서산사(곡성): 서산사는 곡성에 위치한 아름다운 사찰로, 조용하고 평화로운 분위기에서 휴식을 즐길 수 있는 곳입니다. 사찰 내부에는 아름다운 불상과 정자가 있어 마음을 힐링시키기에 안성맞춤입니다. 또한, 주변에는 자연 경관이 아름다운 산과 계곡이 있어 산책이나 등산을 즐길 수도 있습니다.\n\n2. 곡성 단군전: 곡성 단군전은 단군 이야기와 고려시대 역사를 배울 수 있는 곳입니다. 역사적인 분위기를 느끼며 휴식을 즐길 수 있으며, 전통적인 건물과 정원이 아름답게 조성되어 있어 사진을 찍기에도 좋습니다. 또한, 주변에는 조용한 산책로와 카페가 있어 휴식을 즐기기에도 좋습니다.\n\n3. 곡성 메타세쿼이아길: 곡성 메타세쿼이아길은 아름다운 메타세쿼이아 숲을 따라 산책할 수 있는 길입니다. 숲 속을 걷는 동안 신선한 공기를 마시며 휴식을 즐길 수 있습니다. 또한, 메타세쿼이아는 자연 친화적인 나무로 알려져 있어 마음을 힐링시키기에 좋습니다. 길 가에는 휴식 공간이 마련되어 있어 쉬어가며 휴식을 즐길 수도 있습니다.', 'blog': ['https://blog.naver.com/gwangungo/222858748187', 'https://blog.naver.com/mynolto/223136264710', 'https://cu153.com/130']}, '문화시설': {'x': [127.3082114814, 127.3082975874, 127.2974151402], 'y': [35.2776556637, 35.2775723415, 35.2804829367], 'img': ['http://tong.visitkorea.or.kr/cms/resource/30/2789630_image2_1.jpg', 'http://tong.visitkorea.or.kr/cms/resource/59/2663259_image2_1.jpg', 'http://tong.visitkorea.or.kr/cms/resource/51/1971051_image2_1.jpg'], 'name': ['섬진강천적곤충관 (섬진강기차마을생태학습관)', '한국초콜릿연구소 뮤지엄 곡성지점', '곡성문화원'], 'gpt_ans': '1. 섬진강천적곤충관 (섬진강기차마을생태학습관): 이곳은 자연과 생태계에 관심이 있는 분들에게 추천하는 장소입니다. 섬진강 기차마을에 위치한 이 생태학습관은 다양한 천적곤충들을 관찰하고 배울 수 있는 곳으로, 자연 속에서 힐링과 휴식을 즐길 수 있습니다.\n\n2. 한국초콜릿연구소 뮤지엄 곡성지점: 이곳은 초콜릿을 사랑하는 분들에게 추천하는 장소입니다. 곡성에 위치한 한국초콜릿연구소 뮤지엄은 다양한 초콜릿 제품을 전시하고 체험할 수 있는 곳으로, 초콜릿의 향기와 맛을 느끼며 힐링과 휴식을 즐길 수 있습니다.\n\n3. 곡성문화원: 이곳은 문화와 예술을 즐기고 싶은 분들에게 추천하는 장소입니다. 곡성에 위치한 곡성문화원은 다양한 전시물과 예술 작품을 감상할 수 있는 공간으로, 예술의 아름다움을 느끼며 힐링과 휴식을 즐길 수 있습니다.', 'blog': ['http://culture.blogsailing.com/81', 'https://blog.naver.com/leelog_/223125280583', 'http://culture.blogsailing.com/1612']}}


# import streamlit as st
# import time


# st.write('page4')
# st.write(st.session_state.data)
# st.write( st.session_state.next_data['trip_name'])

# def true():
    
# st.button('a', key='but_a', on_click=disable, args=(False,))








# print(test)
# print(st.session_state.testing)
# print(instagram_make)



# import streamlit as st


# import streamlit as st
# import streamlit.components.v1 as components

# placeholder = st.empty()

# with placeholder.container():
#     components.html(
#         """
#         <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script> 
#         <lottie-player src="https://lottie.host/85573875-9faa-4c85-a54e-afbb969a83d6/763ZtSvSup.json" background="transparent" speed="1" style="width: 100%; height: 150%;" loop autoplay></lottie-player>
#         """,
#         height=1000,
#     )
# from streamlit_lottie import st_lottie

# with st_lottie("https://lottie.host/e7f0ded6-5de9-4257-9edb-236c5edd5697/9KTdjwEp9d.json"):
#     time.sleep(5)
# import time
# import requests
# from streamlit_lottie import st_lottie
# from streamlit_lottie import st_lottie_spinner
# import streamlit as st
# from streamlit_lottie import st_lottie
# from streamlit_lottie import st_lottie_spinner


# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()


# lottie_url_hello = "https://lottie.host/e7f0ded6-5de9-4257-9edb-236c5edd5697/9KTdjwEp9d.json"
# # lottie_url_download = "https://assets4.lottiefiles.com/private_files/lf30_t26law.json"
# lottie_hello = load_lottieurl(lottie_url_hello)
# # lottie_download = load_lottieurl(lottie_url_download)


# st_lottie(lottie_hello, key="hello")

# if st.button("Download"):
#     with st_lottie_spinner(lottie_download, key="download"):
#         time.sleep(5)
#     st.balloons()
# #You can check .empty documentation
# placeholder = st.empty()

# with placeholder.container():
#     st.title("Try")
#     btn = st.button("try")

# #If btn is pressed or True
# if btn:
#     #This would empty everything inside the container
#     placeholder.empty()
#     st.write('zz')
    


# button_b = st.button('b', key='but_b', on_click=disable, args=(True,))
# button_c = 
#st.number_input('여행 가능한 최대 거리를 입력해주세요. (km단위)',min_value=5, max_value=100, step=1,key='road')


# import time
# import requests
# from streamlit_lottie import st_lottie_spinner
# import streamlit as st
# from streamlit_lottie import st_lottie_spinner

# def load_lottieurl(url: str):
#     r = requests.get(url)
#     if r.status_code != 200:
#         return None
#     return r.json()

# lottie_url = "https://assets5.lottiefiles.com/packages/lf20_V9t630.json"
# lottie_json = load_lottieurl(lottie_url)

# with st_lottie_spinner(lottie_json):
#     time.sleep(5)
#     st.write('ㅈㅈㅈ')
    # st.balloons()