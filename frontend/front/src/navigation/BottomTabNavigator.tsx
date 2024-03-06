// src/navigation/BottomTabNavigator.tsx
import * as React from 'react';
import { Ionicons } from "@expo/vector-icons";
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Tab1Screen from '../screens/Tab1/Tab1Screen';
import Tab2Screen from '../screens/Tab2/Tab2Screen';
import Tab3Screen from '../screens/Tab3/Tab3Screen';
import Profile from '../screens/Tab3/Profile';
import ProfileEdit from '../screens/Tab3/ProfileEdit';


const Tab = createBottomTabNavigator();
const Tab3Stack = createStackNavigator();


function Tab3StackNavigator() {
  return (
    <Tab3Stack.Navigator screenOptions = {{ headerShown: false }}>
      <Tab3Stack.Screen name="Tab3Screen" component={Tab3Screen} />
      <Tab3Stack.Screen name="Profile" component={Profile} />
      <Tab3Stack.Screen name="ProfileEdit" component={ProfileEdit} />
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
        },
        headerShown: false,
      }}
      initialRouteName="Home"  >
      <Tab.Screen name="기록" component={Tab1Screen} options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="document-sharp" size={18} color={color} style={{ marginTop: 15}}/>
          ),
        }}/>
      <Tab.Screen name="스트리밍" component={Tab2Screen}  options={({ route }) => ({
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="videocam-sharp" size={18} color={color} style={{ marginTop: 15 }}/>
          ),
        })}
        />
      <Tab.Screen name="설정" component={Tab3StackNavigator} options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="settings-sharp" size={18} color={color} style={{ marginTop: 15 }}/>
          ),
        }}/>

    </Tab.Navigator>
  );
}
