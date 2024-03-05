import React, { useEffect } from 'react';
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

type LogDetailScreenRouteProp = RouteProp<RootStackParamList, 'LogDetailScreen'>;
type LogDetailScreenNavigationProp = NavigationProp<RootStackParamList, 'LogDetailScreen'>;
type LogDetailScreenProps = {
    route: LogDetailScreenRouteProp;
    navigation: LogDetailScreenNavigationProp;
  };

const { width, height } = Dimensions.get("screen");


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


    useEffect(() => {
      // This will handle the hardware back press
      const backAction = () => {
        Alert.alert('Hold on!', 'Are you sure you want to go back?', [
          {
            text: 'Cancel',
            onPress: () => null,
            style: 'cancel',
          },
          { text: 'YES', onPress: () => navigation.goBack() },
        ]);
        return true;
      };
  
      // Add back handler event listener
      const backHandler = BackHandler.addEventListener('hardwareBackPress', backAction);
  
      // Remove event listener on cleanup
      return () => backHandler.remove();
    }, [navigation]);

    return (
      <View style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerText}>CCTV 상세 화면</Text>
        <Switch
          trackColor={{ false: "#767577", true: "#81b0ff" }}
          thumbColor={isHighAlert ? "#f5dd4b" : "#f4f3f4"}
          onValueChange={toggleSwitch}
          value={isHighAlert}
        />
      </View>
      <View style={styles.cctvContainer}>
        {/* CCTV 영상 배치 */}
      </View>
      <View style={styles.details}>
        <Text style={styles.detailText}>이름: {cctv_name}</Text>
        <Text style={styles.detailText}>일시: {anomaly_create_time}</Text>
        <Text style={styles.detailText}>이상확률: {anomaly_score}</Text>
        <View style={styles.middle}>
          <TouchableOpacity style={styles.feedback_button}>
            <Text style={styles.buttonText}>피드백 남기기</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.delete_button}>
            <Text style={styles.buttonText}>기록 삭제하기</Text>
          </TouchableOpacity>
        </View>
      </View>
      <View style={styles.footer}>
        <TouchableOpacity 
          style={styles.button}
          onPress={() => {navigation.goBack();
        }}>
          <Text style={styles.buttonText}>이전</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.button}>
          <Text style={styles.buttonText}>저장</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.not_button}>
        </TouchableOpacity>
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
      fontWeight: 'bold',
      fontFamily: 'SG',
    },
    cctvContainer: {
      flex: 1,
      // CCTV 이미지를 위한 스타일
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
      justifyContent: 'space-around',
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
      backgroundColor: argonTheme.COLORS.SUCCESS,
      borderRadius: 5,
      flex: 1,
      marginHorizontal: 10,
    },
    delete_button: {
      padding: 10,
      backgroundColor: argonTheme.COLORS.LABEL,
      borderRadius: 5,
      flex: 1,
      marginHorizontal: 10,
    },    
    button: {
      padding: 10,
      backgroundColor: '#007bff',
      borderRadius: 5,
    },
    not_button: {
      padding: 10,
      borderRadius: 5,
    },
    buttonText: {
      color: '#fff',
      fontSize: 16,
      fontFamily: 'NGB',
    },
  });