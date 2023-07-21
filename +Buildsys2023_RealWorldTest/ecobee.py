import cohort
from time import sleep
import random
from datetime import datetime

HEAT_HOLD_TEMP = 680

ecobee = cohort.Ecobee()

# while(True):
    # curr_temp = ecobee.getData().get('runtime').get("actualTemperature")
    # get outdoor temp
    # TODOs::
    # opt_temp = calc_opt_setpoint(curr_tem, outdoor_temp)
    # ecobee.setHold(HEAT_HOLD_TEMP, opt_temp)

    # # Setting random setpoint
    # opt_temp = round(random.random() * 70 + 710)
    # ecobee.setHold(HEAT_HOLD_TEMP, opt_temp)
    # print(opt_temp)
    # # current date time printing
    # now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)

    # sleep(15 * 60)
# print(ecobee.getData().get('runtime').get("actualTemperature"))

print(ecobee.getHistorical(datetime(2023,6,27),datetime(2023,6,30)))




