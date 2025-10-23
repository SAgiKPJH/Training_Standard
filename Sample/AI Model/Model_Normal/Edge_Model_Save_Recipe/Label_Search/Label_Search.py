import os



step = 'COF'
step = step[:1] + '%' + step[1:]


print(step)

try :
    #path = f'd:/Foundry_ADC/Model_Change/{step}/src/'
    path = f'D:/LBSEMICON/5.LB_Models/Client_Recipe_Division/Noavtek/Cla/Cla_Dataset'
    print(os.listdir(path))
except :
    try :
        path = f'e:/data/modelcheckpoint/완료/전송완료/{step}/src/'
        print(os.listdir(path))
    except :
        try :
            path = f'e:/data/modelcheckpoint/{step}/src/'
            print(os.listdir(path))
        except :
            try :
                path = f'e:/foundry_adc/완료/{step}/src/'
                print(os.listdir(path))
            except :
                path = f'e:/foundry_adc/{step}/src/'
                print(os.listdir(path))


try :
    path = f'e:/foundry_adc/om/{step}/'
    print("om_list = ",os.listdir(path))
except :
    print("해당 스텝은 OM 미적용 입니다.")        


