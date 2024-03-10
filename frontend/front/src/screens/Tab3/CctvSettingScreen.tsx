import React, { useState, useEffect, useContext } from 'react'
import {
  View,
  StyleSheet,
  FlatList,
  Modal,
  TouchableOpacity,
  TextInput,
  ImageBackground,
  Dimensions
} from 'react-native'
import { Text } from 'galio-framework'
import { Images, argonTheme } from '../../constants'
import { NavigationProp } from '@react-navigation/native'
import { RootStackParamList } from '../../navigation/RootStackNavigator'
import { UserContext } from '../../UserContext'

interface Cctvlist {
  cctv_id: number
  cctv_name: string
  cctv_url: string
}

const { width, height } = Dimensions.get("screen");

type Props = {
  navigation: NavigationProp<RootStackParamList, 'CctvSettingScreen'>
}

export default function CctvSettingScreen({ navigation }: Props) {
  const { user } = useContext(UserContext)
  const [Cctvlists, setCctvlists] = useState<Cctvlist[]>([])
  const [registerModalVisible, setRegisterModalVisible] = useState(false)
  const [editModalVisible, setEditModalVisible] = useState(false)
  const [deleteModalVisible, setDeleteModalVisible] = useState(false)
  const [newCctvName, setNewCctvName] = useState('')
  const [newCctvUrl, setNewCctvUrl] = useState('')
  const [cctvId, setCctvId] = useState<number | null>(null)

  const fetchCctvList = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.142:30016/api/v0/settings/cctv_list_lookup?member_id=${user}`,
        {
          method: 'GET',
          headers: { accept: 'application/json' },
        }
      )
      const data = await response.json()
      if (response.ok) {
        setCctvlists(data.result)
      } else {
        console.error('API 호출에 실패했습니다:', data)
      }
    } catch (error) {
      console.error('API 호출 중 예외가 발생했습니다:', error)
    }
  }
  useEffect(() => {
    fetchCctvList()
  }, [])

  const cctvRegister = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.142:30016/api/v0/settings/cctv_register?member_id=${user}&cctv_name=${newCctvName}&cctv_url=${newCctvUrl}`,
        {
          method: 'POST',
          headers: { accept: 'application/json' },
        }
      )
      const data = await response.json()
      if (response.ok) {
        setRegisterModalVisible(false)
        if (data.isSuccess) {
          fetchCctvList()
        }
      } else {
        console.error('API 호출에 실패했습니다:', data)
      }
    } catch (error) {
      console.error('API 호출 중 예외가 발생했습니다:', error)
    }
  }
  const cctvEditer = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.142:30016/api/v0/settings/cctv_edit?cctv_id=${cctvId}&cctv_name=${newCctvName}&cctv_url=${newCctvUrl}`,
        {
          method: 'POST',
          headers: { accept: 'application/json' },
        }
      )
      const data = await response.json()
      setNewCctvName('')
      setNewCctvUrl('')
      setCctvId(null)
      if (response.ok) {
        setEditModalVisible(false)
        if (data.isSuccess) {
          fetchCctvList()
        }
      } else {
        console.error('API 호출에 실패했습니다:', data)
      }
    } catch (error) {
      console.error('API 호출 중 예외가 발생했습니다:', error)
    }
  }
  const cctvDelete = async () => {
    try {
      const response = await fetch(
        `http://10.28.224.142:30016/api/v0/settings/cctv_delete?cctv_id=${cctvId}`,
        {
          method: 'DELETE',
          headers: { accept: 'application/json' },
        }
      )
      const data = await response.json()

      if (response.ok) {
        setCctvId(null)
        setDeleteModalVisible(false)
        if (data.isSuccess) {
          fetchCctvList()
        }
      } else {
        console.error('API 호출에 실패했습니다:', await response.json())
      }
    } catch (error) {
      console.error('API 호출 중 예외가 발생했습니다:', error)
    }
  }

  const renderItem = ({ item }: { item: Cctvlist }) => (
    <View style={styles.item}>
      <View style={styles.item_header}>
        <Text style={styles.itemHeaderText}>{item.cctv_name}</Text>
        <Text style={styles.urlText} numberOfLines={1} ellipsizeMode="tail">
          {item.cctv_url}
        </Text>
      </View>
      <View style={styles.buttons}>
        <TouchableOpacity
          style={styles.feedback_button}
          onPress={() => {
            setEditModalVisible(true)
            setCctvId(item.cctv_id)
            setNewCctvName(item.cctv_name)
            setNewCctvUrl(item.cctv_url)
          }}
        >
          <Text style={styles.buttonText}>수정</Text>
        </TouchableOpacity>
        <Modal
          animationType="fade"
          transparent={true}
          visible={editModalVisible}
          onRequestClose={() => {
            setEditModalVisible(!editModalVisible)
          }}
        >
          <View style={styles.centeredView}>
            <View style={styles.modalView}>
              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={() => {
                  setEditModalVisible(false)
                  setCctvId(null)
                  setNewCctvName('')
                  setNewCctvUrl('')
                }}
              >
                <Text style={{ fontSize: 20, fontWeight: 'bold' }}>X</Text>
              </TouchableOpacity>

              <View style={{ flexDirection: 'column' }}>
                <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                  <Text style={[styles.textModal, {width: '30%'} ]}>CCTV 이름</Text>
                  <TextInput
                    value={newCctvName}
                    onChangeText={setNewCctvName}
                    style={styles.modalTextInput}
                  />
                </View>
                <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                  <Text style={[styles.textModal, {width: '30%'} ]}>URL</Text>
                  <TextInput
                    value={newCctvUrl}
                    onChangeText={setNewCctvUrl}
                    style={styles.modalTextInput}
                  />
                </View>
              </View>

              <TouchableOpacity
                style={styles.modalRegisterButton}
                onPress={() => {
                  cctvEditer()
                }}
              >
                <Text style={styles.modalButtonText}>저장하기</Text>
              </TouchableOpacity>
            </View>
          </View>
        </Modal>
        <TouchableOpacity
          style={styles.delete_button}
          onPress={() => {
            setCctvId(item.cctv_id)
            setDeleteModalVisible(true)
          }}
        >
          <Text style={styles.buttonText}>삭제</Text>
        </TouchableOpacity>
        <Modal
          animationType="fade"
          transparent={true}
          visible={deleteModalVisible}
          onRequestClose={() => {
            setCctvId(null)
            setDeleteModalVisible(false)
          }}
        >
          <View style={styles.centeredView}>
            <View style={styles.modalView}>
              <TouchableOpacity
                style={styles.modalCloseButton}
                onPress={() => {
                  setCctvId(null)
                  setDeleteModalVisible(false)
                }}
              >
                <Text style={{ fontSize: 20, fontWeight: 'bold' }}>X</Text>
              </TouchableOpacity>

              <View style={{ flexDirection: 'column', alignItems: 'center' }}>
                <Text style={styles.textModal}>정말 삭제하시겠습니까?</Text>
                <View
                  style={{
                    flexDirection: 'row',
                    alignItems: 'stretch',
                    justifyContent: 'center',
                  }}
                >
                  <TouchableOpacity
                    style={styles.modalDeleteButtonYes}
                    onPress={() => {
                      cctvDelete()
                    }}
                  >
                    <Text style={styles.buttonText}>예</Text>
                  </TouchableOpacity>
                  <TouchableOpacity
                    style={styles.modalDeleteButtonNo}
                    onPress={() => {
                      setCctvId(null)
                      setDeleteModalVisible(false)
                    }}
                  >
                    <Text style={styles.buttonText}>아니요</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </View>
          </View>
        </Modal>
      </View>
    </View>
  )

  return (
    <ImageBackground
    source={Images.Onboarding}
    style={{ width, height, zIndex: 1 }}
    >
    <View style={styles.container}>
      <FlatList
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={[styles.headerText, {color: 'white'}]}>CCTV 세팅</Text>
            <TouchableOpacity
              style={styles.addButton}
              onPress={() => {
                setRegisterModalVisible(true)
                setNewCctvName('')
                setNewCctvUrl('')
              }}
            >
              <Text style={styles.addButtonText}>+</Text>
            </TouchableOpacity>

            <Modal
              animationType="fade"
              transparent={true}
              visible={registerModalVisible}
              onRequestClose={() => {
                setRegisterModalVisible(!registerModalVisible)
              }}
            >
              <View style={styles.centeredView}>
                <View style={styles.modalView}>
                  <TouchableOpacity
                    style={styles.modalCloseButton}
                    onPress={() =>
                      setRegisterModalVisible(!registerModalVisible)
                    }
                  >
                    <Text style={{ fontSize: 20, fontWeight: 'bold' }}>X</Text>
                  </TouchableOpacity>

                  <View style={{ flexDirection: 'column' }}>
                    <View
                      style={{ flexDirection: 'row', alignItems: 'center' }}
                    >
                      <Text style={[styles.textModal, {width: '30%'} ]}>CCTV 이름</Text>
                      <TextInput
                        value={newCctvName}
                        onChangeText={setNewCctvName}
                        style={styles.modalTextInput}
                      />
                    </View>
                    <View
                      style={{ flexDirection: 'row', alignItems: 'center' }}
                    >
                      <Text style={[styles.textModal, {width: '30%'} ]}>URL</Text>
                      <TextInput
                        value={newCctvUrl}
                        onChangeText={setNewCctvUrl}
                        style={styles.modalTextInput}
                      />
                    </View>
                  </View>

                  <TouchableOpacity
                    style={styles.modalRegisterButton}
                    onPress={() => {
                      cctvRegister()
                    }}
                  >
                    <Text style={styles.modalButtonText}>등록</Text>
                  </TouchableOpacity>
                </View>
              </View>
            </Modal>
          </View>
        }
        data={Cctvlists}
        renderItem={renderItem}
        keyExtractor={(item) => item.cctv_id.toString()}
        style={{ flex: 1 }}
      />
    </View>
    </ImageBackground>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 20,
  },
  item_header: {
    flexDirection: 'column',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: 20,
    flex: 2,
  },
  headerText: {
    fontSize: 20,
    fontWeight: 'bold',
    fontFamily: 'SG',
  },
  itemHeaderText: {
    fontSize: 20,
    fontWeight: 'bold',
    fontFamily: 'NGB',
  },
  urlText: {
    fontSize: 16,
    marginVertical: 4,
    fontFamily: 'NGB',
    flexShrink: 1,
  },
  buttons: {
    paddingVertical: 10,
    flexDirection: 'column',
    justifyContent: 'space-around',
    flex: 1,
  },
  feedback_button: {
    padding: 10,
    marginVertical: 3,
    backgroundColor: argonTheme.COLORS.SUCCESS,
    borderRadius: 5,
    flex: 1,
    marginHorizontal: 10,
  },
  delete_button: {
    padding: 10,
    marginVertical: 3,
    backgroundColor: argonTheme.COLORS.LABEL,
    borderRadius: 5,
    flex: 1,
    marginHorizontal: 10,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    alignContent: 'center',
    textAlign: 'center',
    fontFamily: 'NGB',
  },
  addButton: {
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: argonTheme.COLORS.PLACEHOLDER,
    borderRadius: 25,
    width: 50,
    height: 50,
  },
  addButtonText: {
    fontSize: 43,
    lineHeight: 50,
    color: argonTheme.COLORS.ICON,
    fontWeight: 'bold',
  },
  item: {
    backgroundColor: '#f0f0f0',
    borderWidth: 1,
    borderColor: '#CCCCCC',
    borderRadius: 10,
    padding: 10,
    marginVertical: 10,
    marginHorizontal: 20,
    alignItems: 'center',
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  centeredView: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 22,
  },
  modalView: {
    margin: 20,
    backgroundColor: 'white',
    borderRadius: 20,
    padding: 35,
    alignItems: 'stretch',
    shadowColor: '#000',
    shadowOffset: {
      width: 0,
      height: 2,
    },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 5,
    width: '80%',
  },
  modalTextInput: {
    height: 40,
    margin: 12,
    borderWidth: 1,
    padding: 10,
    borderColor: argonTheme.COLORS.SUCCESS,
    borderRadius: 10,
    width: '70%',
    fontFamily: 'NGB',
  },
  modalCloseButton: {
    alignSelf: 'flex-end',
  },
  modalRegisterButton: {
    marginTop: 10,
    backgroundColor: 'green',
    borderRadius: 20,
    height: 50,
    justifyContent: 'center',
    width: '100%',
  },
  modalDeleteButtonYes: {
    marginTop: 30,
    borderRadius: 20,
    height: 50,
    alignContent: 'center',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'tomato',
    width: '30%',
    padding: 10,
    marginVertical: 3,
    marginHorizontal: 10,
    flex: 1,
  },
  modalDeleteButtonNo: {
    marginTop: 30,
    borderRadius: 20,
    height: 50,
    paddingHorizontal: 20,
    justifyContent: 'center',
    backgroundColor: 'green',
    alignItems: 'center',
    width: '30%',
    padding: 10,
    marginVertical: 3,
    marginHorizontal: 10,
    flex: 1,
  },
  modalButtonText: {
    color: '#fff',
    fontSize: 16,
    textAlign: 'center',
    fontFamily: 'NGB',
  },
  textModal: {
    fontSize: 16,
    marginVertical: 4,
    fontFamily: 'NGB',
  },
})