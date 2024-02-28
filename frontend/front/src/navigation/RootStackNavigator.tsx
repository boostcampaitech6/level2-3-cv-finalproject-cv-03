// src/navigation/RootStackNavigator.tsx
import * as React from 'react';
import { Image } from 'react-native';
import { createStackNavigator } from '@react-navigation/stack';
import SignInScreen from '../screens/SignIn/SignInScreen';
import SignUpScreen from '../screens/SignUp/SignUpScreen';
import BottomTabNavigator from './BottomTabNavigator';
import Tab1Screen from '../screens/Tab1/Tab1Screen';
import Tab2Screen from '../screens/Tab2/Tab2Screen';
import Tab3Screen from '../screens/Tab3/Tab3Screen';
import { TouchableOpacity } from 'react-native';
import { useNavigation } from '@react-navigation/native';



const Stack = createStackNavigator();

export default function RootStackNavigator() {
  const navigation = useNavigation();

  return (
    <Stack.Navigator initialRouteName="SignIn" screenOptions = {{ headerShown: false }}>
      <Stack.Screen name="SignIn" component={SignInScreen} />
      <Stack.Screen name="SignUp" component={SignUpScreen} />
      <Stack.Screen 
        name="Home"
        component={BottomTabNavigator}
        options={{ 
          headerLeft: () => (
            // <TouchableOpacity onPress={() => navigation.navigate('SignInScreen')}>
              <Image source={require('../../assets/Logo.png')} style={{ width: 50, height: 50 }} />
            // </TouchableOpacity>
          ), 
          headerShown: true,
          headerTitle: '',
        }}
      />
      <Stack.Screen name="Tab1Screen" component={Tab1Screen} />
      <Stack.Screen name="Tab2Screen" component={Tab2Screen} />
      <Stack.Screen name="Tab3Screen" component={Tab3Screen} />
    </Stack.Navigator>
  );
}
