from database import *
from flask import Flask, request, redirect, render_template, Response, send_file
import json
from flask_cors import CORS, cross_origin
from flask_mail import Mail, Message
from secrets import token_urlsafe

from PIL import Image
import numpy as np

app = Flask(__name__)
cors = CORS(app, support_credentials=True) # , resources={r"/*": {"origins": "*"}})
app.config['SECRET_KEY'] = "a;lfkdsjaflksdj"
# app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'savetfp.no.reply@gmail.com'
app.config['MAIL_PASSWORD'] = 'savetfpemail'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail = Mail(app)

DEFAULT_PATTERN_JSON = json.dumps({'colorGrid': [["#e3e3e3"]], 'height': 1, 'width': 1, 'stitchWidth': 20, 'stitchHeight':20, 'colors':[]})

from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
@cross_origin()
def home():
    return json.dumps({"accessToken": "...",  "refreshToken": "..."})


@app.route('/users/authenticate',  methods=['POST', 'GET'])
@cross_origin()
def authenticate():
    form = request.get_json()
    user = get_user(form['username'])
    if user != None and user.verify_password(form["password"]):
        set_user_token(user.id, token_urlsafe(16))
        return json.dumps( {'id': user.id,
                            'username': user.username,
                            'token': user.token})
    else:
        return Response(json.dumps({"message":"Invalid login."}), status=401)

@app.route('/users',  methods=['POST', 'GET'])
@cross_origin()
def users():
    # check token
    form = request.get_json()
    if check_token(form['id'], form['token']):
        return json.dumps( [{'id': 'user.id',
                            'username': 'user.username',
                            'firstName': 'user.firstName',
                            'lastName': 'user.lastName',
                            'password': 'password'}])
    return Response(json.dumps({"message":"Invalid token."}), status=401)

@app.route('/patterns',  methods=['POST', 'GET'])
@cross_origin()
def patterns():
    # check token
    form = request.get_json()
    if check_token(form['id'], form['token']):
        return json.dumps([dictPattern(pattern) for pattern in get_users_patterns(form['id'])])
    return Response(json.dumps({"message":"Invalid token."}), status=401)

@app.route('/users/signup',  methods=['POST', 'GET'])
@cross_origin()
def signup():
    #check that username isn't taken
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
    form = request.form.to_dict()
    print("new pattern form",form, request.get_json(), request.form.to_dict())
    if check_token(form['id'], form['token']):
        name =  form['name']
        if name == "":
            name = "Untitled"
        elif get_pattern_user_name(get_user_id(form['id']), name) != None:
            return Response(json.dumps({"message":"Pattern name taken"}), status=400)
        pattern = add_pattern(form['id'], name, DEFAULT_PATTERN_JSON)
        savePatternImg(DEFAULT_PATTERN_JSON, pattern.id)
        # if file uploaded, then save the image
        imageWidth = 1
        imageHeight = 1
        if 'file' in request.files:
            f = request.files['file']
            if f and allowed_file(f.filename):
                extension = f.filename.rsplit('.', 1)[1].lower()
                f.save("upload{}.{}".format(pattern.id,extension))
                add_upload(pattern, f.filename)
                image = Image.open(pattern.upload_filename)
                imageWidth, imageHeight = image.size

        return json.dumps({'id':pattern.id, 'imageWidth':imageWidth, 'imageHeight':imageHeight})
    return Response(json.dumps({"message":"Invalid token."}), status=400)


@app.route('/pattern',  methods=['POST', 'GET'])
@cross_origin()
def pattern():
    # check token
    form = request.get_json()
    if check_token(form['id'], form['token']):
        pattern = get_pattern(form['patternId'])
        if pattern != None:
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
    print("saving pattern", form['id'], form['patternId'])
    if check_token(form['id'], form['token']):
        pattern = get_pattern(form['patternId'])
        if pattern != None:
            if str(pattern.user_id) == form['id']:
                update_pattern(form['patternId'], form['pattern'])
                savePatternImg(form['pattern'], form['patternId'])
                return Response(json.dumps({"message":"Pattern saved."}), status=200)
            return Response(json.dumps({"message":"Not your pattern."}), status=400)
        return Response(json.dumps({"message":"Pattern doesn't exist."}), status=400)
    return Response(json.dumps({"message":"Invalid token."}), status=401)

def dictPattern(pattern):
    return {"id":pattern.id, "name":pattern.name, "pattern":pattern.pattern_json, "timestamp":pattern.timestamp}

def savePatternImg(pattern_json, pattern_id):
    pattern = json.loads(pattern_json)
    nparray = colorGridToNP(pattern['colorGrid']).astype(np.uint8)
    img = Image.fromarray(nparray, 'RGB')
    img = toThumbnail(img)
    img.save('thumbnail{}.png'.format(pattern_id))

def colorGridToNP(colorGrid):
    for y in range(len(colorGrid)):
        row = colorGrid[y]
        for x in range(len(row)):
            row[x] = hexToList(row[x])
    return np.array(colorGrid)

# pattern = get_pattern(22)
# # with open('squad.json', 'w') as f:
# #     json.dump(pattern.pattern_json, f)
# add_pattern(2, 'squadHat', pattern.pattern_json)

def npToColorGrid(data):
    H,W,C = data.shape
    colorGrid = []
    colors = set()
    for h in range(H):
        row = []
        for w in range(W):
            row.append(listToHex(data[h,w].tolist()))
            colors.add(listToHex(data[h,w].tolist()))
        colorGrid.append(row)
    return colorGrid, list(colors)

def hexToList(hexCode):
    h = hexCode.lstrip('#')
    return [int(h[i:i+2], 16) for i in (0, 2, 4)]

def listToHex(rgbList):
    return '#%02x%02x%02x' % tuple(rgbList)

@app.route('/get_image/<pattern_id>')
@cross_origin()
def get_image(pattern_id):
    try:
        filename = 'thumbnail{}.png'.format(pattern_id)
        return send_file(filename, mimetype='image/gif')
    except:
        return send_file("error.png", mimetype='image/gif')

@app.route('/get_upload_image/<pattern_id>')
@cross_origin()
def get_upload(pattern_id):
    # try:
    pattern = get_pattern(pattern_id)
    if pattern != None:
        filename = pattern.upload_filename
        return send_file(filename, mimetype='image/gif')
    # except:
    else:
        return send_file("error.png", mimetype='image/gif')

@app.route('/get_pixelated_image/<pattern_id>')
@cross_origin()
def get_pixelated(pattern_id):
    # try:
    try:
        filename = 'pixelated'+pattern_id+".png"
        return send_file(filename, mimetype='image/gif')
    except:
        return send_file("error.png", mimetype='image/gif')

# todo - I want to be able to make copies of a saved pattern
# upload - route to save image - need a new model table
# pixelate - given image filename and desired width and height and number of colors n, resize to that and then back up
#     also need to determine best n colors and map each pixel to closest match
#     want colors that have high variety but good representation.
#     easy to get the color frequency - go by most frequent colors? But wouldn't allow for best variety
#     seems like a clustering problem - could try an ml cluster - measure success as average distance to a representative cluster member - kmeans
#     or heuristic method - find frequencies - 
# use pixelate when saving colorGrids!

from sklearn.cluster import KMeans
from numpy import asarray

def color_reduce(image_array, n_colors):
    H,W,C = image_array.shape
    X = image_array.reshape(H*W, 3)
    kmeans = KMeans(n_clusters=n_colors, init='k-means++')
    prediction = kmeans.fit_predict(X)
    colors = kmeans.cluster_centers_
    # now convert those predictions into the colors
    return np.array([[colors[prediction[h*W+w]] for w in range(W)] for h in range(H)]).astype(np.uint8)

# todo make authentication wrapper
@app.route('/pixelize',  methods=['POST', 'GET'])
@cross_origin()
def pixelize():
    form = request.get_json()
    pattern_id = form['patternId']
    if check_token(form['id'], form['token']):
        pattern = get_pattern(pattern_id)
        if pattern != None:
            if str(pattern.user_id) == form['id']:
                # get the uploaded image and then pixelate it
                filename = pattern.upload_filename # 'upload'+pattern_id+".png"
                image = Image.open(filename)

                image = image.resize((int(form['width']), int(form['height'])), resample=0)
                data = asarray(image)
                if data.shape[2] == 4: # remove 4th layer if necessary
                    data = data[:,:,:-1]
                reduced_np = color_reduce(data, form['nColors'])
                image = Image.fromarray(reduced_np, 'RGB')
                image.save('pixelated{}.png'.format(pattern_id))

                return Response(json.dumps({"message":"image pixelated."}), status=200)
            return Response(json.dumps({"message":"Not your pattern."}), status=400)
        return Response(json.dumps({"message":"Pattern doesn't exist."}), status=400)
    return Response(json.dumps({"message":"Invalid token."}), status=401)
    
@app.route('/makeImagePattern',  methods=['POST', 'GET'])
@cross_origin()
def make_image_pattern():
    form = request.get_json()
    pattern_id = form['patternId']
    if check_token(form['id'], form['token']):
        pattern = get_pattern(pattern_id)
        if pattern != None:
            if str(pattern.user_id) == form['id']:
                image = Image.open("pixelated{}.png".format(pattern.id))
                data = asarray(image)
                color_grid, colors = npToColorGrid(data)

                pattern_json = json.dumps({'colorGrid':color_grid, 'height':len(color_grid), 'width':len(color_grid[0]), 'stitchWidth': 20, 'stitchHeight':20, 'colors':colors})
                update_pattern(form['patternId'], pattern_json)
                # savePatternImg(pattern_json, form['patternId'])
                return json.dumps({'id':pattern.id})

            return Response(json.dumps({"message":"Not your pattern."}), status=400)
        return Response(json.dumps({"message":"Pattern doesn't exist."}), status=400)
    return Response(json.dumps({"message":"Invalid token."}), status=401)

THUMBNAIL_HEIGHT = 200
THUMBNAIL_WIDTH = 200

def toThumbnail(image):
    # todo rescale width if it is the smaller of the two
    width, height = image.size
    if height < width:
        newHeight = THUMBNAIL_HEIGHT
        size = (newHeight*width//height,newHeight)
    else:
        newWidth = THUMBNAIL_WIDTH
        size = (newWidth,newWidth*height//width)
    image = image.resize(size, resample=0)
    image = image.crop((0,0,THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT))
    return image




@app.route('/uploadimg/<pattern_id>',  methods=['POST', 'GET'])
@cross_origin()
def upload_image(pattern_id):
    form = request.form.to_dict()
    if check_token(form['id'], form['token']):
        pattern = get_pattern(pattern_id)
        if pattern != None:
            if str(pattern.user_id) == form['id']:
                if 'file' in request.files:
                    f = request.files['file']
                    if f and allowed_file(f.filename):
                        extension = f.filename.rsplit('.', 1)[1].lower()
                        f.save("upload{}.{}".format(pattern.id,extension))
                        add_upload(pattern, f.filename)
                        return Response(json.dumps({"message":"image uploaded."}), status=200)
                return Response(json.dumps({"message":"Upload file failed."}), status=400)
            return Response(json.dumps({"message":"Not your pattern."}), status=400)
        return Response(json.dumps({"message":"Pattern doesn't exist."}), status=400)
    return Response(json.dumps({"message":"Invalid token."}), status=401)



@app.route('/email',  methods=['POST', 'GET'])
@cross_origin()
def email():
    form = request.get_json()
    print("form",request.form,request.get_json())
    msg = Message('SaveTFP Care Package', sender = 'savetfp.no.reply@gmail.com', recipients = [form['to_email']+"@mit.edu"])
    msg.body = form['body_email']+'\n\nPlease enjoy these virtual goodies: '+" ".join(form['inventory_email'])+'\n\nFrom '+form['from_email']
    mail.send(msg)
    return 'sent'
   


if __name__ == '__main__':
    app.run(debug=True)
