import React, { useState, useEffect, useContext } from 'react';
import { View, StyleSheet, FlatList, Switch, TouchableOpacity, Alert, BackHandler } from 'react-native';
import { Text } from "galio-framework";
import { Images, argonTheme } from "../../constants";
import { NavigationProp } from '@react-navigation/native';
import { RootStackParamList } from '../../navigation/RootStackNavigator';
import { UserContext } from '../../UserContext';

interface Cctvlist {
  cctv_id: number,
  cctv_name: string,
  cctv_url: string,
}

type Props = {
  navigation: NavigationProp<RootStackParamList, 'CctvSettingScreen'>;
};

export default function CctvSettingScreen({ navigation }: Props) {
  const { user } = useContext(UserContext);
  const [Cctvlists, setCctvlists] = useState<Cctvlist[]>([]);

  useEffect(() => {
    const fetchAnomalyEvents = async () => {
      try {
        const response = await fetch(`http://10.28.224.142:30016/api/v0/settings/cctv_list_lookup?member_id=${user}`, {
          method: "GET",
          headers: { 'accept': 'application/json' },
          });
        console.log('receving data...');
        const data = await response.json();
        console.log(response.ok)
  
        if (response.ok) {
          console.log(data.isSuccess);
          console.log(data.result);
          setCctvlists(data.result);
        } else {
          console.error('API 호출에 실패했습니다:', data);
        }
      } catch (error) {
        console.error('API 호출 중 예외가 발생했습니다:', error);
      }
    };
    fetchAnomalyEvents();
  }, []);

  const renderItem = ({ item }: { item: Cctvlist }) => (
    <View style={styles.item}>
      <View style={styles.item_header}>
        <Text style={styles.itemHeaderText}>{item.cctv_name}</Text>
        <Text style={styles.urlText} numberOfLines={1} ellipsizeMode='tail'>
          {item.cctv_url}
        </Text>
      </View>
      <View style={styles.buttons}>
        <TouchableOpacity style={styles.feedback_button}>
          <Text style={styles.buttonText}>수정</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.delete_button}>
          <Text style={styles.buttonText}>삭제</Text>
        </TouchableOpacity>
      </View>
    </View>
  );

    return (
      <View style={styles.container}>
        <FlatList
          ListHeaderComponent={
            <View style={styles.header}>
            <Text style={styles.headerText}>CCTV 세팅</Text>
            <TouchableOpacity
              style={styles.addButton}
              onPress={() => {
                // TODO: Implement add CCTV functionality
              }}
            >
              <Text style={styles.addButtonText}>+</Text>
            </TouchableOpacity>
          </View>
          }
          data={Cctvlists}
          renderItem={renderItem}
          keyExtractor={item => item.cctv_id.toString()}
          style={{ flex: 1 }}
        />
      </View>
  );};
  
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
    },
    item_header: {
      flexDirection: 'column',
      justifyContent: 'space-between',
      alignItems: 'flex-start',
      padding: 20,
      flex: 2
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
      flex:1
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
      height: 50
    },
    addButtonText: {
      fontSize: 43,
      lineHeight: 50,
      color: argonTheme.COLORS.ICON,
      fontWeight: 'bold'
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
  });