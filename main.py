import cv2 as c

#################################################
cap = c.VideoCapture('videos/human_pass_1.mp4')
frameWait = 30
bgSub = c.createBackgroundSubtractorMOG2()
kernel = c.getStructuringElement(c.MORPH_ELLIPSE, (5, 5))
humans = []
# parâmetros
sensibility = 2000
line = 150
offset = 30
xy1 = (20, line)
xy2 = (300, line)
# resultados
passed = 0
join = 0
out = 0
#################################################


def get_sqr_center(x, y, w, h):
    return x+w//2, y+h//2


while True:
    status, img = cap.read()
    if not status:
        break

    imgGray = c.cvtColor(img, c.COLOR_BGR2GRAY)
    bgMask = bgSub.apply(imgGray)
    thStatus, th = c.threshold(bgMask, 200, 255, c.THRESH_BINARY)
    opMask = c.morphologyEx(th, c.MORPH_OPEN, kernel, iterations=2)
    dlMask = c.dilate(opMask, kernel, iterations=5)
    clMask = c.morphologyEx(dlMask, c.MORPH_CLOSE, kernel, iterations=8)
    c.line(img, xy1, xy2, (0, 255, 0), 3)
    c.line(img, (xy1[0], line+offset), (xy2[0], line+offset), (0, 255, 0), 1)
    c.line(img, (xy1[0], line-offset), (xy2[0], line-offset), (0, 255, 0), 1)

    conts, hier = c.findContours(clMask, c.RETR_TREE, c.CHAIN_APPROX_SIMPLE)

    idC = 0
    for cont in conts:
        x, y, w, h = c.boundingRect(cont)
        area = c.contourArea(cont)
        if area > sensibility:
            center = get_sqr_center(x, y, w, h)
            c.circle(img, center, 4, (255, 255, 0), c.FILLED)
            c.rectangle(img, (x, y), (x+w, y+h), (255, 255, 0), 2)
            c.putText(img, str(idC), (x+5, y+15), c.FONT_ITALIC, 0.5, (0, 0, 255), 2)

            if len(humans) <= idC:
                humans.append([])
            if line-offset < center[1] < line+offset:
                humans[idC].append(center)
            else:
                humans[idC].clear()
            idC += 1
    if len(conts) == 0:
        humans.clear()
    else:
        for human in humans:
            for (cW, l) in enumerate(human):
                if human[cW - 1][1] < line < l[1]:
                    human.clear()
                    out += 1
                    passed += 1
                    c.line(img, xy1, xy2, (0, 255, 255), 5)
                    continue
                if l[1] < line < human[cW - 1][1]:
                    human.clear()
                    join += 1
                    passed += 1
                    c.line(img, xy1, xy2, (0, 0, 255), 5)
                    continue
                if cW > 0:
                    c.line(img, human[cW-1], l, (0, 0, 255), 1)

    if c.waitKey(frameWait) & 0xFF == ord('q'):
        break

    c.putText(img, 'Passaram:' + str(passed), (10, 20), c.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    c.putText(img, 'Entraram:' + str(join), (10, 40), c.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    c.putText(img, 'Sairam:' + str(out), (10, 60), c.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    c.imshow('viewport', img)
    c.imshow('cv', dlMask)

cap.release()
c.destroyAllWindows()
