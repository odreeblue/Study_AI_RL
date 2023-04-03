
# 2. 이 코드에서는 namedtuple을 사용함
# named tuple을 사용하면 키-값 쌍 형태로 값을 저장할 수 있음
# 그리고 키를 필드명으로 값에 접근 할 수 있어 편리함

from Environment import Environment
#Tr = namedtuple('tr', ('name_a', 'value_b'))
#Tr_object = Tr('이름A',100) # 출력 : tr(name_a='이름A', value_b=100)
#print(Tr_object.name_a)  # 출력 : 100
#print(Tr_object.name_a)  # 출력 : 이름A


# 3. 상수 정의
GAMMA  = 0.99 # 시간 할인율
MAX_STEPS =200 #  1에피소드 당 최대 단계 수
NUM_EPISODES = 10000 # 최대 에피소드 수
NUM_STATES = 2 # 태스크의 상태 변수 개수 : x position, z position
NUM_ACTIONS = 4 # 태스크의 행동 가짓 수 : 위, 아래, 오른쪽, 왼쪽
# 실행 엔트리 포인트
cartpole_env = Environment(GAMMA, MAX_STEPS, NUM_EPISODES,NUM_STATES,NUM_ACTIONS)
cartpole_env.run()