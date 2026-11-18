# Day23 / 실습 04 / 응용 / mentee
# 점검용 SQL 문자열을 만들고, 조회 결과와 설치 오류 메시지를 해석합니다.
from day23kit.mathops import cosine_distance

# BEGIN D23-04-I01
def sql_server_info():
    """
    TODO D23-04-I01 · sql_server_info
        pgAdmin에서 실행할 서버 버전·현재 DB 조회 SQL을 작성하세요.
        반드시 SELECT version()과 current_database()를 포함하세요.
        DB 변경이나 DROP/CREATE DATABASE는 하지 않습니다.
    """
    # TODO D23-04-I01: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 서버 버전, 현재 접속한 데이터베이스 이름과 사용자 이름을 조회합니다.
    return "SELECT version(); SELECT current_database(), current_user();"
# END D23-04-I01

# BEGIN D23-04-I02
def sql_extension_state():
    """
    TODO D23-04-I02 · sql_extension_state
        vector 확장의 제공 여부와 설치 여부를 확인하는 SQL 문자열을 작성하세요.
        pg_available_extensions, installed_version, WHERE name='vector' 사용.
    """
    # TODO D23-04-I02: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 서버가 제공하는 확장 중 vector만 조회합니다.
    # default_version은 기본 설치 버전이고, installed_version은 현재 DB에 설치된 버전입니다.
    return "SELECT name, default_version, installed_version From pg_available_extensions WHERE name='vector';"
# END D23-04-I02

# BEGIN D23-04-I03
def sql_distance_probe():
    """
    TODO D23-04-I03 · sql_distance_probe
        SQL에서 3차원 벡터 [1,0,0]과 [0,1,0]의 Cosine distance를 계산하세요.
        문자열 반환. CAST는 ::vector, 연산자는 <=>를 사용합니다.
        예상값 1.0이며 384차원 E5 출력이 아닙니다.
    """
    # TODO D23-04-I03: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # ::vector는 문자열을 벡터 타입으로 바꾸고, <=>는 코사인 거리를 계산합니다.
    # 두 예제 벡터는 직각이므로 코사인 유사도가 0이고 거리(1 - 유사도)는 1.0입니다.
    return "SELECT '[1,0,0]'::vector <=> '[0,1,0]'::vector AS cosine_distance;"
# END D23-04-I03

# BEGIN D23-04-I04
def check_extension_row(row):
    """
    TODO D23-04-I04 · check_extension_row
        fetchone()으로 읽은 (name, default_version, installed_version) 행을 판정하세요.
        None이면 unavailable, installed_version이 None이면 available_not_installed,
        설치됐으면 installed 및 version을 반환합니다.
    """
    # TODO D23-04-I04: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 조회된 행이 없으면 서버가 vector 확장을 제공하지 않는 상태입니다.
    if row is None:
        return {'state':'unavailable', 'version':None}

    # 인덱스는 0부터 시작하므로 row[2]는 세 번째 열인 installed_version입니다.
    installed_version = row[2]

    # 조회 행은 있지만 설치 버전이 비어 있으면 현재 DB에서 아직 활성화하지 않은 상태입니다.
    if installed_version is None:
        return {'state':'available_not_installed', 'version':None}

    return {'state':'installed', 'version':row[2]}
# END D23-04-I04

# BEGIN D23-04-I05
def classify_install_error(message):
    """
    TODO D23-04-I05 · classify_install_error
        대표 오류를 실패 위치(compile, server_files, db_permission,
        db_extension, python_env, other)로 분류하세요.
        키나 비밀번호가 들어 있는 예외 원문은 저장하지 않습니다.
    """
    # TODO D23-04-I05: 위 입력·출력 계약에 맞게 직접 구현합니다.
    # 대소문자 차이를 없애고 위에서부터 대표 문구를 찾습니다.
    # 처음 일치한 분류 이름만 반환하므로 오류 메시지 원문은 결과에 들어가지 않습니다.
    msg=message.lower()
    if 'nmake' in msg or 'postgres.h' in msg:return 'compile'
    if 'vector.control' in msg:return 'server_files'
    if 'permission denied' in msg or 'must be superuser' in msg:return 'db_permission'
    if 'type "vector" does not exist' in msg:return 'db_extension'
    if 'no module named' in msg:return 'python_env'
    return 'other'  # 알려진 문구가 하나도 없는 경우입니다.
# END D23-04-I05
