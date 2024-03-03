import React from 'react';
import { View, Text } from 'react-native';
import { RouteProp } from '@react-navigation/native';
import { RootStackParamList } from '../../navigation/RootStackNavigator';

type LogDetailScreenRouteProp = RouteProp<RootStackParamList, 'LogDetailScreen'>;
type LogDetailScreenProps = {
    route: LogDetailScreenRouteProp;
  };


export default function CCTVDetailScreen(props: LogDetailScreenProps) {
    const { route } = props;
    const { log_id, 
      anomaly_create_time,
      anomaly_save_path,
      cctv_id,
      cctv_name,
      cctv_url} = route.params;
    return (
      <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
        <Text>CCTVDetailScreen</Text>
        <Text>{log_id}</Text>
        <Text>{anomaly_create_time}</Text>
        <Text>{anomaly_save_path}</Text>
        <Text>{cctv_id}</Text>
        <Text>{cctv_name}</Text>
        <Text>{cctv_url}</Text>
      </View>
    );
  };