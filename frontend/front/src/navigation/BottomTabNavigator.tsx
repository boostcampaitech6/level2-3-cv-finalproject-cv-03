// src/navigation/BottomTabNavigator.tsx
import * as React from 'react';
import { Ionicons } from "@expo/vector-icons";
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Tab1Screen from '../screens/Tab1/Tab1Screen';
import Tab2Screen from '../screens/Tab2/Tab2Screen';
import Tab3Screen from '../screens/Tab3/Tab3Screen';
import LogDetailScreen from '../screens/Tab1/LogDetailScreen';
import Profile from '../screens/Tab3/Profile';
import ProfileEdit from '../screens/Tab3/ProfileEdit';
import CctvSettingScreen from '../screens/Tab3/CctvSettingScreen';
import AlarmSettingScreen from '../screens/Tab3/AlarmSettingScreen';
import Alarm from '../screens/Tab3/Alarm';
import AlarmEdit from '../screens/Tab3/AlarmEdit';

const Tab = createBottomTabNavigator();
const Tab1Stack = createStackNavigator();
const Tab3Stack = createStackNavigator();


function Tab1StackNavigator() {
  return (
    <Tab1Stack.Navigator screenOptions = {{ headerShown: false }}>
      <Tab1Stack.Screen name="Tab1Screen" component={Tab1Screen} />
      <Tab1Stack.Screen name="LogDetailScreen" component={LogDetailScreen} />
    </Tab1Stack.Navigator>
  );
}
function Tab3StackNavigator() {
  return (
    <Tab3Stack.Navigator screenOptions = {{ headerShown: false }}>
      <Tab3Stack.Screen name="Tab3Screen" component={Tab3Screen} />
      <Tab3Stack.Screen name="Profile" component={Profile} />
      <Tab3Stack.Screen name="Alarm" component={Alarm} />
      <Tab3Stack.Screen name="AlarmEdit" component={AlarmEdit} />
      <Tab3Stack.Screen name="ProfileEdit" component={ProfileEdit} />
      <Tab3Stack.Screen name="CctvSettingScreen" component={CctvSettingScreen} />
      <Tab3Stack.Screen name="AlarmSettingScreen" component={AlarmSettingScreen} />
    </Tab3Stack.Navigator>
  );
}

export default function BottomTabNavigator() {
  return (
    <Tab.Navigator 
      screenOptions={{
        tabBarActiveTintColor: '#6B24AA', // Replace with your active color
        tabBarInactiveTintColor: '#84898c', // Replace with your inactive color
        tabBarStyle: {
          backgroundColor: '#DAD5F2', // Replace with your background color
        },
        tabBarLabelStyle: {
          fontSize: 12, // Replace with your size
          fontFamily: 'SGB', // Replace with your font family
          marginBottom: 5,
        },
        headerShown: false,
      }}
      initialRouteName="Home"  >
      <Tab.Screen name="기록" component={Tab1StackNavigator} options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="document-sharp" size={18} color={color} style={{ marginTop: 0}}/>
          ),
        }}/>
      <Tab.Screen name="스트리밍" component={Tab2Screen}  options={({ route }) => ({
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="videocam-sharp" size={18} color={color} style={{ marginTop: 0 }}/>
          ),
        })}
        />
      <Tab.Screen name="설정" component={Tab3StackNavigator} options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="settings-sharp" size={18} color={color} style={{ marginTop: 0 }}/>
          ),
        }}/>

    </Tab.Navigator>
  );
}
