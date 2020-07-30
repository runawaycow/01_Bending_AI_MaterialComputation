#_*_ coding:utf-8 _*_
# import cv2.cv as cv
import cv2
import numpy as np
from skimage import morphology, data, color

#识别图中所有白色连通域，再将所有小的连通域改成黑色
def denoising(image):
    #print("desnoising")
    white = []
    image=image/255
    copy_image = image.copy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] > 0 and copy_image[i][j] != -1:
                whiteque = [(i,j)]
                temp_result = []
                while len(whiteque) > 0:
                    _i,_j = whiteque.pop()
                    temp_result.append((_i,_j))
                    copy_image[_i][_j] = -1
                    next = [(_i-1,_j),(_i+1,_j),(_i,_j-1),(_i,_j+1)]
                    for a,b in next:
                        if a >= 0 and b >= 0 and a < len(image) and b < len(image[0]) \
                                and image[a][b] > 0 and copy_image[a][b] != -1:
                            whiteque.append((a,b))
                white.append(temp_result)
                # print(temp_result)
    # print(white)
    for w in white:
        min_b = len(image[0])
        max_b = 0
        min_a = len(image)
        max_a = 0
        for a,b in w:
            min_b = min(b, min_b)
            max_b = max(b, max_b)
            min_a = min(a, min_a)
            max_a = max(a, max_a)
        # print(min_b,max_b,min_a,max_a)
        if max_b - min_b < 100 and max_a - min_a <80:
            for a,b in w:
                image[a][b] = 0
    return image * 255

def expandConverse(image):
    record = [[0]*len(image[0]) for _ in range(len(image))]
    temp = []
    for i in range(len(image)):
        if image[i][0] == 0:
            record[i][0] = 1
            temp.append((i,0))
        if image[i][len(image[0])-1] == 0:
            record[i][len(image[0])-1] = 1
            temp.append((i,len(image[0])-1))
    for i in range(len(image[0])):
        if image[0][i] == 0:
            record[0][i] = 1
            temp.append((0,i))
        if image[len(image)-1][i] == 0:
            record[len(image)-1][i] = 1
            temp.append((len(image)-1,i))
    while len(temp) > 0:
        x,y = temp.pop()
        for a,b in [(x-1,y),(x+1,y),(x,y-1),(x,y+1),(x-1,y-1),(x-1,y+1),(x+1,y-1),(x+1,y+1)]:
            if a < len(image) and b < len(image[0]) and record[a][b] == 0 and image[a][b] == 0:
                temp.append((a,b))
                record[a][b] = 1
    for i in range(len(record)):
        for j in range(len(record[0])):
            if record[i][j] == 0:
                image[i][j] = 255
    return image

def inToOut(image):
    #print("inToOut")
    temp = []
    threshold = 4
    for i in range(threshold,len(image)-threshold):
        for j in range(threshold,len(image[0])-threshold):
            count = 0
            for a,b in [(1,1),(1,2),(1,0),(1,-2),(1,-1),(2,-1),(0,-1),(-2,-1),(-1,-1),
                        (-1,-2),(-1,0),(-2,-1),(-1,1),(-1,2),(0,1),(2,1)]:
                for c in range(1, threshold):
                    a, b = a*c, b*c
                    if  max(abs(a), abs(b))>threshold:
                        break
                    elif image[i+a][j+b] == 255:
                        count += 1
                        break
            if count > 8:
                temp.append((i,j))
    for i,j in temp:
        image[i][j] = 255
    return image
def rawSkeloton(image):
    skeleton = morphology.skeletonize(image/255)
    new_skeleton = [[0] * len(skeleton[0]) for _ in range(len(skeleton))]
    for i in range(len(skeleton)):
        for j in range(len(skeleton[0])):
            if skeleton[i][j]: new_skeleton[i][j] = 255
    new_skeleton = np.array(new_skeleton)
    return new_skeleton
def optimizeSkeloton(image):
    #强制从右下角开始找点
    image0Len = len(image[0])
    curves = [[]]
    for i in range(len(image)):
        find = False
        for j in range(200):
            if image[len(image)-i-1][image0Len-j-1] == 255:
                find = True
                break
        if find:
            curves[0].append((len(image)-i-1,image0Len-j-1))
            break
    # print(curves)
    #从右下角白线开始往上找，并记录是否分叉
    cross = set()
    crossStart = set()#记录曲线分叉之后的第一个点
    neighbotPt = [[0,-1],[-1,-1],[-1,0],[-1,1],[0,1],[1,1],[1,0],[1,-1]]
    for curve in curves:
        needtoEnd = True
        while needtoEnd:
            i,j = curve[-1]
            tempNext = []
            for a, b in neighbotPt:
                ia,jb = i+a,j+b
                test = image[ia][jb]
                if image[ia][jb] == 255 and (ia,jb) not in crossStart and (ia,jb) not in cross:
                    if len(curve) == 1 or (len(curve)>1 and not(ia == curve[-2][0] and jb == curve[-2][1])):
                        tempNext.append((ia,jb))
            if len(tempNext) == 0:
                needtoEnd = False
                break
            nextPt = tempNext.pop()
            while len(tempNext) > 0:
                cross.add((i,j))
                crossStart.add(nextPt)
                curves.append(curve[:])
                temp = tempNext.pop()
                crossStart.add(temp)
                curves[-1].append(temp)
            curve.append(nextPt)
    # print (curves)
    # print (len(curves))
    curve = curves[0]

    for c in curves:
        lencur = len(curve)
        lenc = len(c)
        if len(c) > len(curve):
            curve = c
    newImage = [[0]*image0Len for _ in range(len(image))]
    # print (curve)
    for i,j in curve:
        newImage[i][j] = 255
    return curve,np.array(newImage,np.uint8)

def findBottom(image,i=240,j=360):
    #从i，j开始向右下找，找到第一个右下角没有白点的点
    for k in range(len(image[0])):
        image[len(image)-1][k] = 0
    flagi = i
    flagj = j
    flagb = 240
    while i < len(image)-1 and j < len(image[0])-1 and image[i][j] != 255:#先向右找到第一个白点
        j += 1
    searchPt = [(0,-1),(1,-1),(1,0),(1,1)]
    search = True
    while i < len(image)-1 and j < len(image[0])-1 and image[i+1][j+1] == 255 and search:#再沿着这个白点向左、左下、下、右下寻找白点
        search = False
        if i > flagb:
            break
        for a,b in searchPt:
            ia,jb = i+a,j+b
            if image[ia][jb] == 255:
                i,j = ia,jb
                search = True
                break
    # print("bottom",i,j)
    #从上面搜索到的点作为起始点开始向上搜索下边界轮廓线
    search = True
    searchPt = [(0,-1),(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1)]#为顺时针方向
    curve = [(i,j)]
    flag = 1
    while search:
        search = False
        for _ in range(3):#顺时针方向搜索三次
            a,b = searchPt[flag]
            ia,jb = i+a,j+b
            if i < len(image)-1 and j < len(image[0])-1 and image[ia][jb] > 100:
                i,j = ia, jb
                search = True
                curve.append((i,j))
                break
            # print(ia,jb)
            flag = nextFlag(flag)
        if ia > flagi and jb < flagj:
            break
        flag = nextFlag(flag - 2)
    newImage = [[0] * len(image[0]) for _ in range(len(image))]
    # print (curve)
    for i, j in curve:
        newImage[i][j] = 255
    return curve, np.array(newImage,np.uint8)

def nextFlag(flag):
    if flag == 7: return 0
    else: return flag + 1


def generateData(mid, bottom, step,start_x = 170 ,start_y= 0):
    # print("mid", len(mid))
    # print("bottom", len(bottom))

    # 找到底线的起始点
    bottomFlag = 0
    for n, (i, j) in enumerate(bottom):
        if i == start_x and j > start_y:
            start_y = max(j, start_y)
            bottomFlag = n
    # print("start", n, start_x, start_y)
    # 向上寻找，每一点的梯度根据前后各m个点来判断。然后一个斜向算1.5，如果总数到step，则选择下一个基准点。
    # 找到基准点之后拟合直线，然后寻找附近的底线上的点。用一个flag，大概记录位置开始搜索。找到垂直线最近的两个点。然后取交点距离并乘二
    # 并记录总长度

    midFlag = 8
    midSelect = []
    bottomSelect = []
    beforeVx, beforeVy = -1, 0#初始化搜索方向（图中看的话是向上）
    tempStep = step - 1
    vectorSelect = []

    for n in range(bottomFlag, len(bottom)):
        # print(len(bottomSelect),len(midSelect))
        if bottom[n][0] > start_x: break #结束条件，如果从右向左找最终x>start_x就结束
        tempStep += abs(bottom[n][0] - bottom[n - 1][0]) + abs(bottom[n][1] - bottom[n - 1][1])
        if tempStep > step - 1:
            tempStep = 0

            fitLineRange = 8
            if n-fitLineRange < 0 or n + fitLineRange >= len(bottom):#边缘的点都跳过
                continue
            # 选中的点
            bottomSelect.append(bottom[n])
            aroundPts = np.array(bottom[n - fitLineRange:n + fitLineRange+1])#取选中的点周围的点
            [vx, vy, _, _] = cv2.fitLine(aroundPts, cv2.DIST_LABEL_PIXEL, 0, 0.01, 0.01)#拟合直线
            x = bottom[n][0]
            y = bottom[n][1]

            if beforeVx * vx + beforeVy * vy < 0:#如果法线方向和直线方向是钝角，直线方向旋转180度
                vx, vy = -vx, -vy
            beforeVx, beforeVy = vx, vy#更新搜索方向
            # 垂线
            vx, vy = vy, -vx
            vectorSelect.append((vx,vy))
            nextMidFlag = midFlag + 1

            bbx, bby = mid[midFlag]
            bnx, bny = mid[nextMidFlag]
            db = sinDis(x, y, vx, vy, bbx, bby)#这个值越大说明中线上的像素点（bbx,bby）偏离法线越远
            dn = sinDis(x, y, vx, vy, bnx, bny)
            if db > dn:#接下来几行是为了确定，在底线选中的点沿法线方向和中线的交点，在中线哪两个像素点之间
                while db > dn:
                    db = dn
                    nextMidFlag += 1
                    bnx, bny = mid[nextMidFlag]
                    dn = sinDis(x, y, vx, vy, bnx, bny)
                nextMidFlag -= 1
                midFlag = nextMidFlag - 1
                bbx, bby = mid[midFlag]
                dn = db
                db = sinDis(x, y, vx, vy, bbx, bby)
            else:
                while db < dn:
                    dn = db
                    midFlag -= 1
                    bbx, bby = mid[midFlag]
                    db = sinDis(x, y, vx, vy, bbx, bby)
                midFlag += 1
                nextMidFlag = midFlag + 1
                db = dn
                bnx, bny = mid[nextMidFlag]
                dn = sinDis(x, y, vx, vy, bnx, bny)
            bbx, bby = mid[midFlag]
            bnx, bny = mid[nextMidFlag]#得到中线上两个点之后，作直线和法线相交，得到交点，即为要得到的中线的点
            resx = bbx * dn / (db + dn) + bnx * db / (db + dn)
            resy = bby * dn / (db + dn) + bny * db / (db + dn)
            midSelect.append((resx, resy))
    #print("len of seq ------>    ",len(list(vectorSelect)),len(bottomSelect),len(midSelect))
    if(len(vectorSelect) > 20):#不希望得到的线长度太短，接下来几行代码是为了平滑midSelect
        bvx,bvy = midSelect[0][0] - bottomSelect[0][0], midSelect[0][1] - bottomSelect[0][1]
        norm = np.linalg.norm((bvx,bvy))
        bvx,bvy = bvx/norm,bvy/norm
        count = 0
        for n in range(1,min(len(bottomSelect),len(midSelect)) - 1):
            vx,vy = midSelect[n][0] - bottomSelect[n][0], midSelect[n][1] - bottomSelect[n][1]
            norm = np.linalg.norm((vx,vy))
            vx,vy = vx/norm, vy/norm
            # print (bvx * vx + bvy * vy)
            if bvx * vx + bvy * vy < 0.95:#如果两个法向量角度太大，说明这个向量可能异常，用左右两个点进行平均
                # print("change",midSelect[n])
                count += 1
                midSelect[n] = ((midSelect[n - 1][0] + midSelect[n + 1][0]) / 2,(midSelect[n - 1][1] + midSelect[n + 1][1]) / 2)
            bvx,bvy = vx,vy
    # print ("-----------------------",count)
        if len(bottomSelect)>len(midSelect):
            bottomSelect=bottomSelect[:len(midSelect)]
        return midSelect, bottomSelect, vectorSelect
    else:
        return [],[],[]

def sinDis(x,y,vx,vy,tx,ty):#计算点（x,y）和点（tx,ty）的距离减去这两个点组成的向量在向量（vx,vy）上的投影长度。
    cosDis = abs(vx*(tx-x) + vy*(ty-y))
    dis = np.linalg.norm((tx-x,ty-y))
    return dis - cosDis

def denoiseUpDown(image):
    for i in range(1,len(image)-1):
        for j in range(1,len(image[0])-1):
            if image[i][j] > 0 and ((image[i-1][j]==0 and image[i+1][j])==0 or (image[i][j-1]==0 and image[i][j+1]) == 0):
                image[i][j] = 0
            if image[i][j] == 0 and ((image[i-1][j]!= 0 and image[i+1][j] != 0) or (image[i][j-1]!=0 and image[i][j+1]!=0)):
                image[i][j] = max(image[i-1][j], image[i][j-1])
    return image

def smooth(image):
    for i in range(1,len(image)-1):
        for j in range(1,len(image[0])-1):
            count0,count1 = 0,0
            for a,b in ([-1,0],[1,0],[0,1],[0,-1]):
                if image[i+a][j+b]==0:
                    count0+=1
                else:
                    count1+=1
            if image[i][j] == 0 and count1 == 3:#如果一个白色像素周围3个都是黑色像素，那么这个像素改为黑色
                # print('000')
                image[i][j] = max(image[i][j-1],image[i][j+1])
            if image[i][j] > 0 and count0 == 3:#如果一个黑色像素周围3个都是白色像素，那么这个像素改为白色
                # print ('111')
                image[i][j] = 0

            if image[i][j] == 0 and ((image[i+1][j-1]>0 and image[i-1][j+1]>0) or (image[i-1][j-1]>0 and image[i+1][j+1]>0)):
                image[i][j] = max(image[i-1][j-1],image[i-1][j+1])
    return image


