from database import *
from flask import Flask, request, redirect, render_template, Response, send_file
import json
from flask_cors import CORS, cross_origin
from secrets import token_urlsafe

from PIL import Image
import numpy as np

app = Flask(__name__)
cors = CORS(app, support_credentials=True)
app.config['SECRET_KEY'] = "a;lfkdsjaflksdj"
app.config['CORS_HEADERS'] = 'Content-Type'

DEFAULT_PATTERN_JSON = json.dumps({'colorGrid': [["#e3e3e3"]], 'height': 1, 'width': 1, 'stitchWidth': 20, 'stitchHeight':20, 'colors':[]})

@app.route('/')
@cross_origin()
def home():
    return json.dumps({"accessToken": "...",  "refreshToken": "..."})


@app.route('/users/authenticate',  methods=['POST', 'GET'])
@cross_origin()
def authenticate():
    print("authernticate", request.get_json())
    form = request.get_json()
    user = get_user(form['username'])
    if user != None and user.verify_password(form["password"]):
        set_user_token(user.id, token_urlsafe(16))
        return json.dumps( {'id': user.id,
                            'username': user.username,
                            'token': user.token})
    else:
        return Response(json.dumps({"message":"Invalid login."}), status=400)

@app.route('/users',  methods=['POST', 'GET'])
@cross_origin()
def users():
    # check token
    print("get users", request.get_json())
    form = request.get_json()
    if check_token(form['id'], form['token']):
        return json.dumps( [{'id': 'user.id',
                            'username': 'user.username',
                            'firstName': 'user.firstName',
                            'lastName': 'user.lastName',
                            'password': 'password'}])
    return Response(json.dumps({"message":"Invalid token."}), status=400)

@app.route('/patterns',  methods=['POST', 'GET'])
@cross_origin()
def patterns():
    # check token
    print("get patterns", request.get_json())
    form = request.get_json()
    if check_token(form['id'], form['token']):
        return json.dumps([dictPattern(pattern) for pattern in get_users_patterns(form['id'])])
    return Response(json.dumps({"message":"Invalid token."}), status=400)

@app.route('/users/signup',  methods=['POST', 'GET'])
@cross_origin()
def signup():
    #check that username isn't taken
    print("signup", request.get_json())
    form = request.get_json()
    user = get_user(form['username'])
    if user == None:
        user = create_user(form['username'],form['password'])
        set_user_token(user.id, token_urlsafe(16))
        return json.dumps( {'id': user.id,
                            'username': user.username,
                            'token': user.token})
    return Response(json.dumps({"message":"Username taken"}), status=400)
    
@app.route('/patterns/new',  methods=['POST', 'GET'])
@cross_origin()
def newPattern():
    # check token
    print("new pattern", request.get_json())
    form = request.get_json()
    if check_token(form['id'], form['token']):
        name =  form['name']
        if get_pattern_user_name(get_user_id(form['id']), name) != None:
            return Response(json.dumps({"message":"Pattern name taken"}), status=400)
        pattern = add_pattern(form['id'], name, DEFAULT_PATTERN_JSON)
        return json.dumps({'id':pattern.id})
    return Response(json.dumps({"message":"Invalid token."}), status=400)


@app.route('/pattern',  methods=['POST', 'GET'])
@cross_origin()
def pattern():
    # check token
    print("get pattern", request.get_json())
    form = request.get_json()
    if check_token(form['id'], form['token']):
        pattern = get_pattern(form['patternId'])
        if pattern != None:
            print(pattern.user_id, form['id'], pattern.id == form['id'], type(pattern.user_id), type(form['id']))
            if str(pattern.user_id) == form['id']:
                return json.dumps(dictPattern(pattern))
            return Response(json.dumps({"message":"Not your pattern."}), status=400)
        return Response(json.dumps({"message":"Pattern doesn't exist."}), status=400)
    return Response(json.dumps({"message":"Invalid token."}), status=400)

@app.route('/savePattern',  methods=['POST', 'GET'])
@cross_origin()
def savePattern():
    # check token
    
    form = request.get_json()
    print("save pattern", form['pattern'])
    if check_token(form['id'], form['token']):
        pattern = get_pattern(form['patternId'])
        if pattern != None:
            print(pattern.user_id, form['id'], str(pattern.user_id) == form['id'], type(pattern.user_id), type(form['id']))
            if str(pattern.user_id) == form['id']:
                update_pattern(form['patternId'], form['pattern'])
                savePatternImg(form['pattern'], form['patternId'])
                return Response(json.dumps({"message":"Pattern saved."}), status=200)
            return Response(json.dumps({"message":"Not your pattern."}), status=400)
        return Response(json.dumps({"message":"Pattern doesn't exist."}), status=400)
    return Response(json.dumps({"message":"Invalid token."}), status=400)

def dictPattern(pattern):
    return {"id":pattern.id, "name":pattern.name, "pattern":pattern.pattern_json, "timestamp":pattern.timestamp}

def savePatternImg(pattern_json, pattern_id):
    pattern = json.loads(pattern_json)
    nparray = colorGridToNP(pattern['colorGrid']).astype(np.uint8)
    img = Image.fromarray(nparray, 'RGB')
    img.save('pattern'+pattern_id+'.png')

def colorGridToNP(colorGrid):
    for y in range(len(colorGrid)):
        row = colorGrid[y]
        for x in range(len(row)):
            row[x] = hexToList(row[x])
    return np.array(colorGrid)

def hexToList(hexCode):
    h = hexCode.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

@app.route('/get_image/<pattern_id>')
@cross_origin()
def get_image(pattern_id):
    print("get image",request.args.get('type'),request)
    try:
        filename = 'pattern'+pattern_id+".png"
        return send_file(filename, mimetype='image/gif')
    except:
        return send_file("error.png", mimetype='image/gif')

if __name__ == '__main__':
    app.run(debug=True)
