# HashTrip

- 기워드 기반 여행추천 사이트
  - instagram과 같이 Tag 기반으로 여행을 추천, 여행 추천이유, 여행 경로 추천 모든 과정을 PDF로 만들어 주면 조금더 여행을 편하게 하지 않을까 라는 생각으로 수행한 프로젝트 입니다.
  - 모든 동작은 Python을 통해서 동작합니다. (streamlit, langchain, openai, deap)

- Test web site : https://hashtrip-lztggxyjxhazaeyrjwzj2w.streamlit.app/
  - 해당 web은 streamlit의 무료 배포를 사용했기때문에 너무 많은 요청 및 빠른 변경을 수행하면 오류가 발생할수 있습니다.
  - (무료 기준 ram 1G, CPU 1G)

&nbsp;
<details>
  <summary><b>Hash Trip 동작 설명</b></summary>
  <div markdown="1">
    <ul>
    </br>
      <li>추천 화면 설명</li>
      <ul>
          <li>기본적으로 모든 키워드는 #서울, #힐링, #여행 과 같이 입력하여 가고싶은 여행지와 여행의 테마를 입력하면 공공API와 KAKAO API를 통해서 정보를 얻고 해당 정보를 vectorDB로 변환합니다.</li>
          <li>변환된 VectorDB와 ChatGPT를 langchain을 통해 연결후 ChatGPT에게 사전에 설정한 추천수 만큼 추천을 수행하게 됩니다.</li>
          <li>추천된 결과는 지도, 추천장소, ChatGPT 장소 추천이유, 추천장소 이미지, 해당 장소관련 블로그 리뷰글 이 추천됩니다.</li>
          </br>
            <details> <summary><b>추천 화면</b></summary> 나중에 gif 들어갈곳 </details>
      </ul>
      
  </br>
    <li>관광지 경로 추천 및 PDF 화면 설명</li>
    <ul>
      <li>모든 결과를 instagram의 feed 형식으로 다시한번 전달하여 사용자가 맘에드는 장소에 대한 최대 여행 가능 거리, 관광지별 선호도 점수를 받아서 유전알고리즘으로 관광지 경로를 추천합니다.</li>
      <li>최종적으로 여태까지의 모든 과정을 PDF로 만들어서 사용자가 다운받아 보실수 있습니다.</li>
          </br>
            <details> <summary><b>관광지 경로 추천 및 PDF 화면</b></summary> 나중에 gif 들어갈곳 </details>
    </ul>

  
  </ul>
  </div>
</details>
