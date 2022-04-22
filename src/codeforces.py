import csv
import requests
import sys
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select

BASE_URL = 'http://codeforces.com/api/'

class Contest:
    def __init__(self, json):
        self.json = json
        self.contest_id = json['id']


class Problem:
    def __init__(self, json):
        self.json = json
        self.tags = json['tags']
        self.name = json['name']


class Submission:
    def __init__(self, json):
        self.json = json


class RanklistRow:
    def __init__(self, json):
        self.json = json
        self.problem_results = json['problemResults']


def get_request(endpoint, params={}):
    time.sleep(0.5)
    return requests.get('{}{}'.format(BASE_URL, endpoint), params=params).json()


def list_contests():
    req = get_request('contest.list', params={'gym': False})
    res = []
    for contest in req['result']:
        res.append(Contest(contest))
    return res


def get_problems():
    req = get_request('problemset.problems')
    res = []
    for problem in req['result']['problems']:
        res.append(Problem(problem))
    return res


def get_standings(contest_id, count=1000):
    req = get_request('contest.standings', params={'contestId': int(contest_id), 'count': count, 'from': 1, 'showUnofficial': True})
    res = []
    problems = []
    if 'result' not in req:
        print(req)
        return None
    for row in req['result']['rows']:
        res.append(RanklistRow(row))
    for problem in req['result']['problems']:
        problems.append(Problem(problem))
    return res, problems




languages = [['Java 11', 'Java 8'], ['Haskell'], ['Kotlin 1.4', 'Kotlin 1.5', 'Kotlin 1.6'], ['Ocaml'], ['Go'], ['GNU C++17', 'GNU C++14']]


def get_problem_submissions(contest_id, problem, language_id=0):
    url = 'https://codeforces.com/problemset/status/{}/problem/{}'.format(contest_id, problem)
    submissions = []
    MAX_SUBMISSIONS = 4
    language = languages[language_id]
    print('getting {}/{}'.format(contest_id, problem))
    for language_name in language:
        if len(submissions) >= MAX_SUBMISSIONS:
            break
        options = Options()
        options.headless = True
        driver_tries = 0
        while driver_tries < 3:
            try:
                driver = webdriver.Chrome(options=options)
                driver.get(url)
                verdict_name = Select(driver.find_element_by_id('verdictName'))
                verdict_name.select_by_visible_text('Accepted')
                language_select = Select(driver.find_element_by_id('programTypeForInvoker'))
                language_select.select_by_visible_text(language_name)
                apply_button = driver.find_element_by_css_selector('input[value=\'Apply\']')
                apply_button.click()
                cookies = driver.get_cookies()
                request_cookies = {}
                for cookie in cookies:
                    if cookie['name'] == 'JSESSIONID':
                        request_cookies['JSESSIONID'] = cookie['value']
                        break
                driver.close()
                break
            except:
                time.sleep(40)
                driver_tries += 1
                continue
        if driver_tries >= 3:
            break
        req = requests.get(url, cookies=request_cookies)
        bs = BeautifulSoup(req.text, features='html.parser')
        for tr in bs.find_all('tr'):
            time.sleep(3)
            tries = 0
            while tries < 3:
                try:
                    cells = tr.find_all('td')
                    if len(cells) == 0:
                        break
                    if 'No items' in cells[0].contents[0]:
                        break
                    href = 'https://codeforces.com' + cells[0].a['href']
                    sub = BeautifulSoup(requests.get(href).text, features='html.parser')
                    lang = cells[4].get_text()
                    status = cells[5].get_text()
                    if status.strip() == 'Accepted':
                        code = sub.find(class_='linenums').contents[0]
                        submissions.append(code)
                        if len(submissions) >= MAX_SUBMISSIONS:
                            break
                    break
                except Exception as e:
                    print(e)
                    print('Failed, going to wait to try again')
                    if tries == 0:
                        time.sleep(60)
                    else:
                        time.sleep(3)
                    tries += 1
            if tries >= 3:
                break
            if len(submissions) >= MAX_SUBMISSIONS:
                break
        if len(submissions) >= MAX_SUBMISSIONS:
            break

    print(len(submissions), 'found')
    return submissions


def get_processed_problems(language):
    try:
        f = open('../data/codeforces_tags_{}_processed.csv'.format(language), 'r')
        processed = set()
        reader = csv.reader(f)
        for problem in reader:
            contest_id = problem[0]
            index = problem[1]
            # print((contest_id, index))
            processed.add((contest_id, index))
        f.close()
        return processed
    except:
        return set()


def get_language_id(language):
    if language == 'java':
        return 0
    elif language == 'haskell':
        return 1
    elif language == 'kotlin':
        return 2
    elif language == 'ocaml':
        return 3
    elif language == 'go':
        return 4
    elif language == 'cpp':
        return 5
    else:
        raise NameError('Unexpected language')



def get_data(language):

    language_id = get_language_id(language)

    processed = get_processed_problems(language)
    print(len(processed))
    f = open('../data/codeforces_tags_{}.csv'.format(language), 'a')
    problem_f = open('../data/codeforces_tags_{}_processed.csv'.format(language), 'a')
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    problem_writer = csv.writer(problem_f, quoting=csv.QUOTE_ALL)
    problems = get_problems()
    print('Got all problems')
    total = 0
    for problem in problems:
        if (str(problem.json['contestId']), problem.json['index']) in processed:
            continue
        tags = ','.join(problem.tags)
        submissions = get_problem_submissions(problem.json['contestId'], problem.json['index'], language_id=language_id)
        for sub in submissions:
            code = sub
            code = code.replace('\n', '\\n')
            code = code.replace('\r', '')
            writer.writerow([code, tags, problem.json['contestId'], problem.json['index']])
        problem_writer.writerow([problem.json['contestId'], problem.json['index']])
        total += 1
        if total % 10 == 0:
            print('{} problems processed'.format(total))
    f.close()
    problem_f.close()
