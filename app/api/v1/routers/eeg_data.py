from fastapi import APIRouter
from app.core.eeg_data_processor import get_eeg_data, get_patient_info, get_data_sec
from fastapi.responses import JSONResponse
eeg_router = APIRouter()


@eeg_router.get('/eegData')
def get_egg_data(start_time: int, end_time: int, montage_type: int):
    print('test')
    result_list = get_eeg_data(start_time, end_time, montage_type)
    if result_list:
        return result_list
    else:
        return JSONResponse(
            status_code=404,
            content={"message": f"Index not found"},
        )


@eeg_router.get('/test')
def patient_id():
    print('test')
    return 'test'


@eeg_router.get('/Time_Information')
def data_sec():
    sec = get_data_sec()
    return sec


@eeg_router.get('/eegPatientInfo/{patient_id}')
def eeg_patient_info(patient_id: str):
    print('id: ', patient_id)

    return get_patient_info(patient_id)


@eeg_router.get('/eegDataQuery/{id}')
def get_eeg_data2(id: str, start_time: int = 0, end_time: int = 0):
    print('id: ', id)
    print('start_time: ', start_time)
    print('end_time: ', end_time)

    return get_eeg_data(start_time, end_time)
