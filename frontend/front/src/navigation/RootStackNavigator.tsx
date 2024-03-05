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

import Onboarding from '../screens/Onboarding/Onboarding';
import Register from '../screens/SignUp/Register';
import Register2 from '../screens/SignUp/Register2';
import Register3 from '../screens/SignUp/Register3';
import Register4 from '../screens/SignUp/Register4';
import Register5 from '../screens/SignUp/Register5';
import Register6 from '../screens/SignUp/Register6';

import Login from '../screens/SignIn/Login';

export type RootStackParamList = {
  SignIn: undefined;
  SignUp: undefined;
  Home: undefined;
  Tab1Screen: undefined;
  Tab2Screen: undefined;
  Tab3Screen: undefined;
  Onboarding: undefined;
  Register: undefined;
  Register2: undefined;
  Register3: undefined;
  Register4: undefined;
  Register5: undefined;
  Register6: undefined;
  Login: undefined;
  LogDetailScreen: {   
    anomaly_create_time: string,
    cctv_id: number,
    anomaly_save_path: string,
    anomaly_delete_yn: boolean,
    log_id: number,
    anomaly_score: number,
    anomaly_feedback: boolean,
    member_id: number,
    cctv_name: string,
    cctv_url: string
  };
};

const Stack = createStackNavigator<RootStackParamList>();

export default function RootStackNavigator() {
  const navigation = useNavigation();

  return (
    <Stack.Navigator initialRouteName="Onboarding" 
      screenOptions = {{ 
        headerShown: false,
        headerTitle: '와치덕',
        headerLeft: () => (
          <Image
            source={require('../../assets/Logo.png')}
            style={{ width: 50, height: 50, resizeMode: 'contain' }}
          />)}}>
      <Stack.Screen name="Onboarding" component={Onboarding} />
      <Stack.Screen name="Register" component={Register} />
      <Stack.Screen name="Register2" component={Register2} />
      <Stack.Screen name="Register3" component={Register3} />
      <Stack.Screen name="Register4" component={Register4} />
      <Stack.Screen name="Register5" component={Register5} />
      <Stack.Screen name="Register6" component={Register6} />
      <Stack.Screen name="Login" component={Login} />

      <Stack.Screen name="SignIn" component={SignInScreen} />
      <Stack.Screen name="SignUp" component={SignUpScreen} />
      <Stack.Screen name="Home" component={BottomTabNavigator}
        options={{ 
          headerLeft: () => (
            // <TouchableOpacity onPress={() => navigation.navigate('SignInScreen')}>
              <Image source={require('../../assets/Logo.png')} style={{ width: 50, height: 50 }} />
            // </TouchableOpacity>
          ), 
          headerShown: true,
          headerTitle: '와치덕',
        }}/>
      <Stack.Screen name="Tab1Screen" component={Tab1Screen} />
      <Stack.Screen name="Tab2Screen" component={Tab2Screen} />
      <Stack.Screen name="Tab3Screen" component={Tab3Screen} />
      <Stack.Screen name="LogDetailScreen" component={LogDetailScreen} />
    </Stack.Navigator>
  );
}