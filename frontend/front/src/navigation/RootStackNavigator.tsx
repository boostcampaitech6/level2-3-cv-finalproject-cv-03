import * as React from 'react';
import { Image } from 'react-native';
import { createStackNavigator } from '@react-navigation/stack';
import SignInScreen from '../screens/SignIn/SignInScreen';
import SignUpScreen from '../screens/SignUp/SignUpScreen';
import BottomTabNavigator from './BottomTabNavigator';
import Tab1Screen from '../screens/Tab1/Tab1Screen';
import Tab2Screen from '../screens/Tab2/Tab2Screen';
import Tab3Screen from '../screens/Tab3/Tab3Screen';
import LogDetailScreen from "../screens/Tab1/LogDetailScreen";
import { useNavigation } from '@react-navigation/native';

export type RootStackParamList = {
  SignIn: undefined;
  SignUp: undefined;
  Home: undefined;
  Tab1Screen: undefined;
  Tab2Screen: undefined;
  Tab3Screen: undefined;
  LogDetailScreen: {   
    log_id: number;
    anomaly_create_time: string;
    anomaly_save_path: string;
    cctv_id: number;
    cctv_name: string;
    cctv_url: string; 
  };
};

const Stack = createStackNavigator<RootStackParamList>();

export default function RootStackNavigator() {
  const navigation = useNavigation();

  return (
    <Stack.Navigator initialRouteName="SignIn" 
      screenOptions = {{ 
        headerShown: true,
        headerTitle: '가디언아이즈',
        headerLeft: () => (
          <Image
            source={require('../../assets/Logo.png')}
            style={{ width: 50, height: 50, resizeMode: 'contain' }}
          />)}}>
    <Stack.Screen name="SignIn" component={SignInScreen} />
    <Stack.Screen name="SignUp" component={SignUpScreen} />
    <Stack.Screen name="Home" component={BottomTabNavigator}/>
    <Stack.Screen name="Tab1Screen" component={Tab1Screen} />
    <Stack.Screen name="Tab2Screen" component={Tab2Screen} />
    <Stack.Screen name="Tab3Screen" component={Tab3Screen} />
    <Stack.Screen name="LogDetailScreen" component={LogDetailScreen} />
    </Stack.Navigator>
  );
}