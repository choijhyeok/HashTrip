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
from streamlit_extras.customize_running import center_running
from streamlit_extras.streaming_write import write
import time
import streamlit_ext as ste
import fitz
from weasyprint.text.fonts import FontConfiguration
from weasyprint import HTML, CSS
import os
plt.rcParams['figure.figsize'] = [15, 8]
st.set_page_config(page_title="HashTrip",initial_sidebar_state="collapsed",layout="wide")



if "sec_number" not in st.session_state:
    st.session_state['sec_number'] = 0
    st.session_state['check_row'] = 0
    st.session_state['package_logs'] = 0
    st.session_state['out_text'] = ''
    st.session_state['data_frame'] = ''
    
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




def make_html(html_string,data,gpt, out_text, data_frame):
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
    html_string += '<p><img src="result.png" referrerpolicy="no-referrer" alt="유전알고리즘 그래프"></p><p>&nbsp;</p>'
    
    for idx, n in enumerate(data_frame):
        sep_str = ''
        for j in range(len(n[1:-2])):
            if j != len(n[1:-2])-1:
                sep_str += str(n[j])
                sep_str += '->'
            else:
                sep_str += str(n[j])
        html_string += f'{idx}번 추천경로 : {sep_str} km : {n[-2]} 선호도총합 : {n[-1]}<br>'

    html_string += '<p>&nbsp;</p>'
    new_out = out_text.replace('\n','<br>')
    html_string +=f'<p>{new_out}</p></div></div></body>'
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

    def __init__(self, maxKM):

        # initialize instance variables:
        self.items = []
        self.maxKm = maxKM
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

def list_in_tuple(data):
    all_list = []
    for i in range(len(data)):
        all_list.append(tuple(data.iloc[i]))

    return all_list



def DEAP_float(data, maxKM):
    check_row = []
    while(True):
        MAX_GENERATIONS = 300
        POPULATION_SIZE = 30
        P_CROSSOVER = 0.9
        P_MUTATION = 0.1
        HALL_OF_FAME_SIZE = 1
        RANDOM_SEED = 42
        random.seed(RANDOM_SEED)
        knapsack = Knapsack01Problem(maxKM)
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


        # print("-- Best Ever Individual = ", best)
        # print("-- Best Ever Fitness = ", best.fitness.values[0])

        # print("-- Knapsack Items --")
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

        # print()
        if MAX_GENERATIONS == best_gen:
            # print('\n 현재 수행결과가 가장좋은 결과입니다.')
            return check_row, package_logs
            break
        else:
            # print('최고의 설정은 best avg : {}, best gen : {} 입니다.'.format(best_avg,best_gen))
            MAX_GENERATIONS = best_gen
            return check_row, package_logs
            break

def product_sep(data,keys):
    all_vars = []
    all_y = []
    all_x = []
    preference_dict = dict()
    
    cnt = 0
    for  i in keys:
        all_vars.append(data[i]['name'])
        all_y.append(data[i]['y'])
        all_x.append(data[i]['x'])
        
        for j in data[i]['name']:
            preference_dict[j] =  st.session_state.data[f'set{cnt}']
            cnt += 1
    
    
    if len(keys) == 2:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1])))
        for idx,i in enumerate(product_list):
            km_float = haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], abs(km_float), preference_dict[all_vars[0][i[0]]] +preference_dict[all_vars[1][i[1]]]))
        return df_list
    
    elif len(keys) == 3:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]],km_float, preference_dict[all_vars[0][i[0]]] +preference_dict[all_vars[1][i[1]]] +preference_dict[all_vars[2][i[2]]]))
        return df_list
    
    elif len(keys) == 4:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]],km_float, preference_dict[all_vars[0][i[0]]] +preference_dict[all_vars[1][i[1]]] +preference_dict[all_vars[2][i[2]]] +preference_dict[all_vars[3][i[3]]]))
        return df_list

    elif len(keys) == 5:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]],km_float, preference_dict[all_vars[0][i[0]]] +preference_dict[all_vars[1][i[1]]] +preference_dict[all_vars[2][i[2]]] +preference_dict[all_vars[3][i[3]]] +preference_dict[all_vars[4][i[4]]]))
        return df_list
    
    elif len(keys) == 6:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4], all_vars[5]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4]), all_vars[5].index(i[5])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km')) + abs(haversine((float(all_y[4][i[4]]), float(all_x[4][i[4]])), (float(all_y[5][i[5]]), float(all_x[5][i[5]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]], all_vars[5][i[5]],km_float, preference_dict[all_vars[0][i[0]]] +preference_dict[all_vars[1][i[1]]] +preference_dict[all_vars[2][i[2]]] +preference_dict[all_vars[3][i[3]]] +preference_dict[all_vars[4][i[4]]] +preference_dict[all_vars[5][i[5]]]))
        return df_list
    
    elif len(keys) == 7:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4], all_vars[5], all_vars[6]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4]), all_vars[5].index(i[5]), all_vars[6].index(i[6])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km')) + abs(haversine((float(all_y[4][i[4]]), float(all_x[4][i[4]])), (float(all_y[5][i[5]]), float(all_x[5][i[5]])), unit = 'km')) + abs(haversine((float(all_y[5][i[5]]), float(all_x[5][i[5]])), (float(all_y[6][i[6]]), float(all_x[6][i[6]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]], all_vars[5][i[5]], all_vars[6][i[6]],km_float, preference_dict[all_vars[0][i[0]]] +preference_dict[all_vars[1][i[1]]] +preference_dict[all_vars[2][i[2]]] +preference_dict[all_vars[3][i[3]]] +preference_dict[all_vars[4][i[4]]] +preference_dict[all_vars[5][i[5]]] +preference_dict[all_vars[6][i[6]]]))
        return df_list
    
    elif len(keys) == 8:
        product_list=[]
        df_list = []
        for i in product(all_vars[0], all_vars[1],all_vars[2],all_vars[3], all_vars[4], all_vars[5], all_vars[6], all_vars[7]):
            product_list.append((all_vars[0].index(i[0]), all_vars[1].index(i[1]), all_vars[2].index(i[2]), all_vars[3].index(i[3]), all_vars[4].index(i[4]), all_vars[5].index(i[5]), all_vars[6].index(i[6]) , all_vars[7].index(i[7])))
        for idx,i in enumerate(product_list):
            km_float = abs(haversine((float(all_y[0][i[0]]), float(all_x[0][i[0]])), (float(all_y[1][i[1]]), float(all_x[1][i[1]])), unit = 'km')) + abs(haversine((float(all_y[1][i[1]]), float(all_x[1][i[1]])), (float(all_y[2][i[2]]), float(all_x[2][i[2]])), unit = 'km')) + abs(haversine((float(all_y[2][i[2]]), float(all_x[2][i[2]])), (float(all_y[3][i[3]]), float(all_x[3][i[3]])), unit = 'km')) + abs(haversine((float(all_y[3][i[3]]), float(all_x[3][i[3]])), (float(all_y[4][i[4]]), float(all_x[4][i[4]])), unit = 'km')) + abs(haversine((float(all_y[4][i[4]]), float(all_x[4][i[4]])), (float(all_y[5][i[5]]), float(all_x[5][i[5]])), unit = 'km')) + abs(haversine((float(all_y[5][i[5]]), float(all_x[5][i[5]])), (float(all_y[6][i[6]]), float(all_x[6][i[6]])), unit = 'km')) + abs(haversine((float(all_y[6][i[6]]), float(all_x[6][i[6]])), (float(all_y[7][i[7]]), float(all_x[7][i[7]])), unit = 'km'))
            df_list.append((idx, all_vars[0][i[0]], all_vars[1][i[1]], all_vars[2][i[2]], all_vars[3][i[3]] , all_vars[4][i[4]], all_vars[5][i[5]], all_vars[6][i[6]] ,all_vars[7][i[7]] ,km_float, preference_dict[all_vars[0][i[0]]] +preference_dict[all_vars[1][i[1]]] +preference_dict[all_vars[2][i[2]]] +preference_dict[all_vars[3][i[3]]] +preference_dict[all_vars[4][i[4]]] +preference_dict[all_vars[5][i[5]]] +preference_dict[all_vars[6][i[6]]] +preference_dict[all_vars[7][i[7]]]))
        return df_list
    
def stream_example(package_logs, check_row, road, df):
    semi_text1 = f'''

    ##### :red[**최종 추천 결과**] 
    
    :총 합이 :red[**{road}**] km를 넘지않고 선호도가 최대인 여행지 조합 추천 {package_logs}, choice rows = {check_row}
    '''
    semi_text2 = f'''
    Hashtrip의 최종 여행의 추천입니다. \n\n
    
    입력된 최대거리 {road}km를 기반으로 유전알고리즘 추천을 했을때 {check_row} 번호의 여행들이 최대거리를 넘지않으면서 최대의 선호도 점수를 기록하는 여행지 입니다. \n
    해당 여행지의 합산 거리, 합산 선호도는  {package_logs} 입니다. \n\n
    
    추천된 조합을 여행에 참고하셔서 즐거운 여행 되시길 바랍니다. 
    '''
    
    # dfi.export(df, 'DF.png', max_cols=-1, max_rows=-1)
    
    
    for word in semi_text1.split():
        yield word + " "
        time.sleep(0.1)
    
    yield df
    
    for word in semi_text2.split():
        yield word + " "
        time.sleep(0.05)
    
    return semi_text2


htmlf = open('template.html')
html_string = htmlf.read()
cssf = open('template.css')
css_string = cssf.read()
data_df = product_sep(st.session_state.pdf_data,list(st.session_state.pdf_data.keys()))
data_df = pd.DataFrame(data_df)



    
with col1:
    seper1, seper2, seper3 = st.columns([120,60,120])
    with seper2:
        st.button('경로기반 추천', disabled=True)
    
    if st.session_state['sec_number']  == 0:
        center_running()
        check_row, package_logs = DEAP_float(data_df.sample(frac=1), st.session_state.data['road'])
        st.session_state['check_row'] = check_row
        st.session_state['package_logs'] = package_logs
    
        st.image('result.png')
        st.divider()
        seper11, seper22, seper33 = st.columns([30,200,30])
        with seper22:
            if st.button('HashTrip 경로기반 추천 결과'):
                df_render = pd.DataFrame(data_df.iloc[check_row][data_df.columns[1:]])
                semi_text2 = write(stream_example(package_logs[-1], list(map(int,st.session_state['check_row'])), st.session_state.data["road"], df_render))
                st.session_state['out_text'] = semi_text2[-1]
                st.session_state['data_frame'] = df_render
                st.session_state['sec_number'] =1
                # string_html = make_html(html_string,st.session_state.pdf_data,st.session_state.gpt,semi_text2[-1],df_render.iloc[list(map(int,st.session_state['check_row']))].values)
                # font_config = FontConfiguration()
                # html = HTML(string=string_html, base_url='.')
                # css = CSS(string=css_string, font_config=font_config)
                # html.write_pdf('HashTrip.pdf', stylesheets=[css], font_config=font_config)
                # st.session_state['sec_number'] =2
    else:
        st.image('result.png')
        st.divider()
        seper11, seper22, seper33 = st.columns([30,200,30])
        with seper22:
            if st.button('HashTrip 경로기반 추천 결과'):
                semi_text2 = write(stream_example(st.session_state['package_logs'][-1], list(map(int,st.session_state['check_row'])), st.session_state.data["road"], pd.DataFrame(data_df.iloc[st.session_state['check_row']][data_df.columns[1:]])))
                st.session_state['out_text'] = semi_text2[-1]
                st.session_state['sec_number'] =2
        
            # st.write('##### :red[**최종 추천 결과**]')
            # st.markdown(f'- 총 합이 :red[**{st.session_state.data["road"]} km**]를 넘지않고 선호도가 최대인 여행지 조합 추천')
            # st.markdown(f'- {package_logs[-1]}, choice rows = {list(map(int,check_row))}')
            # st.dataframe(data_df.iloc[check_row][data_df.columns[1:]])
            
# else:
#     with col1:
#         st.image('result.png')
#         st.divider()
#         seper11, seper22, seper33 = st.columns([30,200,30])
#         with seper22:
#             if st.button('HashTrip 경로기반 추천 결과'):
#                 semi_text2 = write(stream_example(st.session_state['package_logs'][-1], list(map(int,st.session_state['check_row'])), st.session_state.data["road"], pd.DataFrame(data_df.iloc[st.session_state['check_row']][data_df.columns[1:]])))
#                 st.session_state['out_text'] = semi_text2[-1]
#                 st.session_state['sec_number'] =2
                
with col2:
    if st.session_state['sec_number'] == 1:
        st.session_state['sec_number']=2
        # sort_check_row = list(map(int,st.session_state['check_row']))
        # st.write(sort_check_row)
        # st.write(st.session_state['out_text'])
        # st.write(st.session_state['data_frame'])
        string_html = make_html(html_string,st.session_state.pdf_data,st.session_state.gpt,st.session_state['out_text'] ,st.session_state['data_frame'].values)
        font_config = FontConfiguration()
        html = HTML(string=string_html, base_url='.')
        css = CSS(string=css_string, font_config=font_config)
        html.write_pdf('HashTrip.pdf', stylesheets=[css], font_config=font_config)
        pdf_doc = fitz.open('HashTrip.pdf')
        for i in pdf_doc:
            pix = i.get_pixmap()
            pix.save(f"page-{i.number}.png" )       
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
            with open('HashTrip.pdf', 'rb') as f:
                st.download_button(':blue[⬇ Download HashTrip PDF]', f, file_name='HashTrip.pdf')


# with col2:
#     if st.session_state['sec_number'] == 1:
#         st.session_state['sec_number'] = 2
#         # string_html = make_html(html_string,st.session_state.pdf_data,st.session_state.gpt,st.session_state['semi_text2'])
#         # font_config = FontConfiguration()
#         # html = HTML(string=string_html, base_url='.')
#         # css = CSS(string=css_string, font_config=font_config)
#         # html.write_pdf('template.pdf', stylesheets=[css], font_config=font_config)
#         # pdf_doc = fitz.open('template.pdf')
#         # for i in pdf_doc:
#         #     pix = i.get_pixmap()
#         #     pix.save("page-%i.png" % i.number)
        
#     seper4, seper5, seper6 = st.columns([120,80,120])
#     seper44, seper55, seper66 = st.columns([30,120,30])
#     seper7, seper8, seper9 = st.columns([120,80,120])   
#     with seper5:
#         st.button('여행스타그램 전체 과정 정리', disabled=True)
#     with seper55:
#         st.write(' ')
#         st.write(' ')
#         st.write(' ')
#         image_list = glob(f'page-*.png')
#         image_list.sort()
#         number = ste.select_slider(
#             '**HashTrip PDF 미리보기**',
#             options=image_list,
#             key="select_slider",
#         )
#         st.image(number, width=600)
    
#     with seper8:
#         with open('template.pdf', 'rb') as f:
#             st.download_button(':blue[⬇ Download HashTrip PDF]', f, file_name='HashTrip.pdf')
