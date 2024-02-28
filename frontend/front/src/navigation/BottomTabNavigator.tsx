// src/navigation/BottomTabNavigator.tsx
import * as React from 'react';
import { Ionicons } from "@expo/vector-icons";
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import Tab1Screen from '../screens/Tab1/Tab1Screen';
import Tab2Screen from '../screens/Tab2/Tab2Screen';
import Tab3Screen from '../screens/Tab3/Tab3Screen';

const Tab = createBottomTabNavigator();

export default function BottomTabNavigator() {
  return (
    <Tab.Navigator initialRouteName="Home"  screenOptions = {{ headerShown: false }}>
      <Tab.Screen name="기록" component={Tab1Screen} options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="document-sharp" size={size} color={color} />
          ),
        }}/>
      <Tab.Screen name="스트리밍" component={Tab2Screen}  options={({ route }) => ({
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="videocam-sharp" size={size} color={color} />
          ),
        })}
        />
      <Tab.Screen name="설정" component={Tab3Screen} options={{
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="settings-sharp" size={size} color={color} />
          ),
        }}/>
    </Tab.Navigator>
  );
}
