import React, { useState, useEffect, useContext  } from 'react';
import {
  View,
  StyleSheet,
  Switch,
  TouchableOpacity,
  Dimensions,
  BackHandler,
  Alert,
} from 'react-native';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../navigation/RootStackNavigator';
import { Block, Text, theme } from "galio-framework";
import { Images, argonTheme } from "../../constants";
import { NavigationProp } from '@react-navigation/native';
import { Overlay } from 'react-native-elements';
import { Button, Input } from "../../components";


type LogDetailScreenRouteProp = RouteProp<RootStackParamList, 'LogDetailScreen'>;
type LogDetailScreenNavigationProp = NavigationProp<RootStackParamList, 'LogDetailScreen'>;
type LogDetailScreenProps = {
    route: LogDetailScreenRouteProp;
    navigation: LogDetailScreenNavigationProp;
  };

export default function CCTVDetailScreen({ route, navigation }: LogDetailScreenProps) {
    const { 
      anomaly_create_time,
      cctv_id,
      anomaly_save_path,
      anomaly_delete_yn,
      log_id,
      anomaly_score,
      anomaly_feedback,
      member_id,
      cctv_name,
      cctv_url
      } = route.params;
    
    const [isHighAlert, setIsHighAlert] = React.useState(false);
    const toggleSwitch = () => setIsHighAlert(previousState => !previousState);
    const [fail, setFail] = React.useState(false);
    const [saveVisible, setSaveVisible] = useState(false);
    const [save2Visible, setSave2Visible] = useState(false);
    const [deleteVisible, setDeleteVisible] = useState(false);
    const [delete2Visible, setDelete2Visible] = useState(false);

    const handleDelete = async () => {
      try {
        const response = await fetch(`http://10.28.224.142:30016/api/v0/cctv/log_delete?log_id=${log_id}`, {

          method: 'DELETE',
          headers: {
            'accept': 'application/json',
          },
          // body: JSON.stringify({ email }),
        });
        // console.log(email)
        const data = await response.json();
        console.log(data);
        if (data.isSuccess) {
          setFail(false)
          navigation.navigate("Tab1Screen");
        }
        else {
          setFail(true)
      }
      } catch (error) {
        console.error('Network error:', error);
      }
    };



    const handleFeedback = async () => {
      try {

        const response = await fetch(`http://10.28.224.142:30016/api/v0/cctv/feedback?log_id=${log_id}&feedback=${1}`, {

          method: 'PUT',
          headers: {
            'accept': 'application/json',
          },
          // body: JSON.stringify({ email }),
        });
        // console.log(email)
        const data = await response.json();
        console.log(data);
        if (data.isSuccess) {
          setFail(false)
        }
        else {
          setFail(true)
      }
      } catch (error) {
        console.error('Network error:', error);
      }
    };

    const handleFeedback2 = async () => {
      try {
        const response = await fetch(`http://10.28.224.142:30016/api/v0/cctv/feedback?log_id=${log_id}&feedback=${0}`, {

          method: 'PUT',
          headers: {
            'accept': 'application/json',
          },
          // body: JSON.stringify({ email }),
        });
        // console.log(email)
        const data = await response.json();
        console.log(data);
        if (data.isSuccess) {
          setFail(false)
        }
        else {
          setFail(true)
      }
      } catch (error) {
        console.error('Network error:', error);
      }
    };

    return (
      <View style={styles.container}>
        <View style={styles.header}>
          <Text style={styles.headerText}>{cctv_name}</Text>
          <Switch
            trackColor={{ false: "#767577", true: "#610C9F" }}
            thumbColor={isHighAlert ? "#DAD5F2" : "#f4f3f4"}
            onValueChange={toggleSwitch}
            value={isHighAlert}
          />
        </View>
        <View style={styles.cctvContainer}>
          {/* CCTV 영상 배치 */}
        </View>
        <View style={styles.details}>
          <Text style={styles.detailText}>일시: {anomaly_create_time}</Text>
          <Text style={styles.detailText}>이상확률: {anomaly_score}</Text>
          <View style={styles.middle}>
            <TouchableOpacity style={styles.feedback_button} onPress={() => setSaveVisible(true)}>
              <Text style={styles.buttonText}>피드백{'\n'}남기기</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.download_button}>
              <Text style={styles.buttonText}>영상{'\n'}다운로드</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.delete_button} onPress={() => setDeleteVisible(true)}>
              <Text style={styles.buttonText}>기록{'\n'}삭제하기</Text>
            </TouchableOpacity>
          </View>
          {fail && (
            <Text style={styles.failText}>작업에 실패했습니다.</Text>
          )}
          <Overlay isVisible={saveVisible} onBackdropPress={() => setSaveVisible(false)}>
            <View style={{ alignItems: 'center', justifyContent: 'center', padding: 40 }}>
              <Text style={styles.poptitle}>이 분석이 맞나요?</Text>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Button style={{marginTop:20, width:100,}} color="success" onPress={() => {
                  handleFeedback();
                  setSaveVisible(false);
                  setSave2Visible(true);
                }} >
                  <Text style={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}>
                    예
                  </Text>
                </Button>
                <Button style={{marginTop:20, width:100,}} color="error" onPress={() => {
                  handleFeedback2();
                  setSaveVisible(false);
                  setSave2Visible(true);
                }} >
                  <Text style={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}>
                    아니오
                  </Text>
                </Button>
              </View>
            </View>
          </Overlay>

          <Overlay isVisible={save2Visible} onBackdropPress={() => setSave2Visible(false)}>
            <View style={{ alignItems: 'center', justifyContent: 'center', padding: 40 }}>
              <Text style={styles.poptitle}>피드백이 반영되었습니다.</Text>                    
            </View>
          </Overlay>

          <Overlay isVisible={deleteVisible} onBackdropPress={() => setDeleteVisible(false)}>
            <View style={{ alignItems: 'center', justifyContent: 'center', padding: 40 }}>
              <Text style={styles.poptitle}>삭제하시겠습니까?</Text>
              <View style={{ flexDirection: 'row', justifyContent: 'space-between' }}>
                <Button style={{marginTop:20, width:100,}} color="success" onPress={() => {
                  handleDelete();
                  setDeleteVisible(false);
                  setDelete2Visible(true);
                }} >
                  <Text style={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}>
                    예
                  </Text>
                </Button>
                <Button style={{marginTop:20, width:100,}} color="error" onPress={() => {
                  setDeleteVisible(false);
                  setDelete2Visible(false);
                }} >
                  <Text style={{ fontSize: 14, color: argonTheme.COLORS.WHITE, fontFamily: 'NGB',}}>
                    아니오
                  </Text>
                </Button>
              </View>
            </View>
          </Overlay>

          <Overlay isVisible={delete2Visible} onBackdropPress={() => setDelete2Visible(false)}>
            <View style={{ alignItems: 'center', justifyContent: 'center', padding: 40 }}>
              <Text style={styles.poptitle}>삭제되었습니다.</Text>                    
            </View>
          </Overlay>
        </View>
      </View>
  );
  };
  
  const styles = StyleSheet.create({
    container: {
      flex: 1,
      backgroundColor: '#f0f0f0',
    },
    header: {
      flexDirection: 'row',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: 20,
      backgroundColor: '#fff',
    },
    headerText: {
      fontSize: 20,
      // fontWeight: 'bold',
      fontFamily: 'C24',
    },
    cctvContainer: {
      flex: 1,
    },
    details: {
      padding: 20,
      backgroundColor: '#fff',
    },
    detailText: {
      fontSize: 16,
      marginVertical: 4,
      fontFamily: 'NGB',
    },
    footer: {
      flexDirection: 'row',
      justifyContent: 'flex-start',
      paddingVertical: 20,
      backgroundColor: '#fff',
      bottom: 0,
    },
    middle: {
      paddingVertical: 20,
      flexDirection: 'row',
      justifyContent: 'space-around',
    },
    feedback_button: {
      padding: 10,
      backgroundColor: "#610C9F",
      borderRadius: 5,
      flex: 1,
      marginHorizontal: 10,
    },
    download_button: {
      padding: 10,
      backgroundColor: "#940B92",
      borderRadius: 5,
      flex: 1,
      marginHorizontal: 10,
    },  
    delete_button: {
      padding: 10,
      backgroundColor: "#DA0C81",
      borderRadius: 5,
      flex: 1,
      marginHorizontal: 10,
    },    
    buttonText: {
      color: '#fff',
      fontSize: 16,
      alignContent: 'center',
      textAlign: 'center',
      fontFamily: 'C24',
    },
    failText: {
      color: argonTheme.COLORS.ERROR,
      fontFamily: 'NGB',
      fontSize: 13,
    },
    poptitle: {
      fontFamily: 'C24',
      marginBottom: 30,
      fontSize: 20,
    },
  });