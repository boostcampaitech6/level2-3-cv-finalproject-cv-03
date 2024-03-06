import React, { useEffect } from 'react';
import { View, StyleSheet, Switch, TouchableOpacity, Alert, BackHandler } from 'react-native';
import { Text } from "galio-framework";
import { Images, argonTheme } from "../../constants";
import { NavigationProp } from '@react-navigation/native';
import { RootStackParamList } from '../../navigation/RootStackNavigator';

type Props = {
  navigation: NavigationProp<RootStackParamList, 'CctvSettingScreen'>;
};

export default function CctvSettingScreen({ navigation }: Props) {
  const [isHighAlert, setIsHighAlert] = React.useState(false);
  const toggleSwitch = () => setIsHighAlert(previousState => !previousState);

  useEffect(() => {
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
        <Text style={styles.detailText}>이름</Text>
        <View style={styles.middle}>
          <TouchableOpacity style={styles.feedback_button}>
            <Text style={styles.buttonText}>피드백 남기기</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.download_button}>
            <Text style={styles.buttonText}>영상 다운로드</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.delete_button}>
            <Text style={styles.buttonText}>기록 삭제하기</Text>
          </TouchableOpacity>
        </View>
      </View>
      <View style={styles.footer}>
        <TouchableOpacity 
          style={styles.back_button}
          onPress={() => {navigation.goBack();
        }}>
          <Text style={styles.buttonText}>이전</Text>
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
      backgroundColor: argonTheme.COLORS.SUCCESS,
      borderRadius: 5,
      flex: 1,
      marginHorizontal: 10,
    },
    download_button: {
      padding: 10,
      backgroundColor: argonTheme.COLORS.GRADIENT_START,
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
    back_button: {
      padding: 10,
      backgroundColor: argonTheme.COLORS.ACTIVE,
      borderRadius: 5,
      width: 120,
      height: 40,
      marginHorizontal: 20,
    },
    buttonText: {
      color: '#fff',
      fontSize: 16,
      alignContent: 'center',
      textAlign: 'center',
      fontFamily: 'NGB',
    },
  });