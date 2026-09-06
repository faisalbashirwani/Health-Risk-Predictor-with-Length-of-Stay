# **Make Predictions**

# Predict the Risk Level of any patient
# Format: [Respiratory_Rate,	Oxygen_Saturation,	O2_Scale,	Systolic_BP,	Heart_Rate,
#           Temperature,	Consciousness('A', 'C', 'P', 'U', 'V'),	On_Oxygen]
# (A = Alert, P = Pain response, C = Confusion, V = Verbal, U = Unresponsive)
# Whether the patient is on supplemental oxygen (0 = No, 1 = Yes).
# Encoding:
#   Consciousness:    {'A':0,'C':1,'P':2, 'U':3, 'V':4}
#   Risk_Level: {'Normal':0,'Low':1,'Medium':2, 'High':3}

# Respiratory_Rate,Oxygen_Saturation,O2_Scale,Systolic_BP,Heart_Rate,Temperature,Consciousness,On_Oxygen,Risk_Level

# gender,dialysisrenalendstage,asthma,irondef,pneum,substancedependence,psychologicaldisordermajor,
#depress,psychother,fibrosisandother,malnutrition,hemo,hematocrit,neutrophils,sodium,glucose,bloodureanitro,
#creatinine,bmi,pulse,respiration,secondarydiagnosisnonicd9,discharged,facid, rcount -> lengthofstay



import numpy as np
import joblib

def predict_risk():
  risk_model_lr = joblib.load('risk_model_lr.pkl')
  risk_model_rf = joblib.load('risk_model_rf.pkl')
  input_data = []
  #input data in input_data as eg; [25,96,1,97,107,37.5,'A',0]
  Format = ['Respiratory_Rate',	'Oxygen_Saturation',	'O2_Scale',	'Systolic_BP',	'Heart_Rate',
          'Temperature',	"Consciousness('A', 'C', 'P', 'U', 'V')",	'On_Oxygen']
  k = 0
  for i in Format:
    if k == 6:
      data = input(f'Enter {i}: ')
    else:
      data = float(input(f'Enter {i}: '))

    input_data.append(data)
    k += 1

  mapping = {'A':0,'C':1,'P':2, 'U':3, 'V':4}
  input_data[6] = mapping[input_data[6]]
  input_array = np.asarray(input_data).reshape(1, -1)

  predicted_risk = risk_model_rf.predict(input_array)

  if predicted_risk == 0:
    print("Predicted Risk Level: Normal")
  elif predicted_risk == 1:
    print("Predicted Risk Level: Low")
  elif predicted_risk == 2:
    print("Predicted Risk Level: Medium")
  else:
    print("Predicted Risk Level: High")
  return predicted_risk

def predict_stay():
  base_model = joblib.load('base_model.pkl')
  format = ['gender F→1, M→0', 'dialysis_renal_endstage 0 or 1', 'asthma 0 or 1', 'irondef 0 or 1', 'pneum 0 or 1', 'substance_dependence 0 or 1', 'psychological_disordermajor 0 or 1',
              'depress 0 or 1', 'psychother 0 or 1', 'fibrosisandother 0 or 1', 'malnutrition 0 or 1', 'hemo 0 or 1','hematocrit float e.g. 11.5', 'neutrophils float e.g. 14.2', 
              'sodium float e.g. 140.3', 'glucose float e.g. 192.4', 'bloodureanitro float e.g. 12.0', 'creatinine float e.g. 1.39', 'bmi float e.g. 30.4', 'pulse  int e.g. 96',
              'respiration float e.g. 6.5', 'secondary_diagnosis_nonicd9 int e.g. 4',    'facid A->0, B->1, C->2, D->3 or E->4', 'revisit count 0, 1, 2, 3, 4 or 5 (use 5 for 5+)']
  input_data = []
  for i in format:
      data = float(input(f'Enter {i}: '))
      input_data.append(data)

  input_array = np.asarray(input_data).reshape(1, -1)
  predicted_stay = base_model.predict(input_array)

  print(f"Predicted Length of Stay: {predicted_stay[0]} days")
  return predicted_stay


#to check health risk
choice_risk = input(f"Enter R to check if patient has Health Risk: ")
if choice_risk == 'R':
  risk = predict_risk()
  if risk == 2 or risk == 3:
    choice_stay = input(f"Enter S to check how much long a patient has to Stay: ")
    if choice_stay == 'S':
      stay = predict_stay()
  else:
    print("bye...")

