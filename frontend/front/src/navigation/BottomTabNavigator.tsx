import React, { useEffect, useContext } from "react";
import { Ionicons } from "@expo/vector-icons";
import { createStackNavigator } from "@react-navigation/stack";
import { createBottomTabNavigator } from "@react-navigation/bottom-tabs";
import Tab1Screen from "../screens/Tab1/Tab1Screen";
import Tab2Screen from "../screens/Tab2/Tab2Screen";
import Tab3Screen from "../screens/Tab3/Tab3Screen";
import LogDetailScreen from "../screens/Tab1/LogDetailScreen";
import Profile from "../screens/Tab3/Profile";
import CctvSettingScreen from "../screens/Tab3/CctvSettingScreen";
import Alarm from "../screens/Tab3/Alarm";
import AlarmEdit from "../screens/Tab3/AlarmEdit";
import { UserContext } from "../UserContext";

import { useLastNotificationResponse } from "expo-notifications";
import { useNavigation } from "@react-navigation/native";

import * as Notifications from "expo-notifications";

Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: false,
  }),
});

const Tab = createBottomTabNavigator();
const Tab1Stack = createStackNavigator();
const Tab3Stack = createStackNavigator();

function Tab1StackNavigator() {
  return (
    <Tab1Stack.Navigator screenOptions={{ headerShown: false }}>
      <Tab1Stack.Screen name="Tab1Screen" component={Tab1Screen} />
      <Tab1Stack.Screen name="LogDetailScreen" component={LogDetailScreen} />
    </Tab1Stack.Navigator>
  );
}
function Tab3StackNavigator() {
  return (
    <Tab3Stack.Navigator screenOptions={{ headerShown: false }}>
      <Tab3Stack.Screen name="Tab3Screen" component={Tab3Screen} />
      <Tab3Stack.Screen name="Profile" component={Profile} />
      <Tab3Stack.Screen name="Alarm" component={Alarm} />
      <Tab3Stack.Screen name="AlarmEdit" component={AlarmEdit} />
      <Tab3Stack.Screen
        name="CctvSettingScreen"
        component={CctvSettingScreen}
      />
    </Tab3Stack.Navigator>
  );
}

export default function BottomTabNavigator() {
  const { user } = useContext(UserContext);
  const navigation = useNavigation();
  const notificationResponse = useLastNotificationResponse();

  // console.log(user)

  useEffect(() => {
    const requestPermissions = async () => {
      const { status } = await Notifications.requestPermissionsAsync();
      if (status !== "granted") {
        alert(
          "No notification permissions. You might want to enable notifications for this app.",
        );
      }
    };

    requestPermissions();
  }, []);

  useEffect(() => {
    if (notificationResponse) {
      navigation.navigate("기록", { screen: "Tab1Screen" });
    }
  }, [notificationResponse]);

  const previousResultRef = React.useRef<number | null>(null);

  React.useEffect(() => {
    const intervalId = setInterval(async () => {
      const response = await fetch(
        `http://10.28.224.201:30438/api/v0/cctv/log_count?member_id=${user}`,
      );
      const result = await response.json();
      console.log("bottom tab navigator");
      console.log(result.result);
      console.log(previousResultRef.current);

      // previousResultRef.current = result.result;
      if (previousResultRef.current === null) {
        previousResultRef.current = result.result;
      } else if (
        previousResultRef.current !== null &&
        result.result > previousResultRef.current
      ) {
        Notifications.scheduleNotificationAsync({
          content: {
            title: "도난 의심 행위 발생",
            body: "확인 바랍니다.",
          },
          trigger: {
            seconds: 1,
          },
        });

        previousResultRef.current = result.result;
      }
    }, 5000);

    return () => clearInterval(intervalId);
  }, []);

  return (
    <Tab.Navigator
      screenOptions={{
        tabBarActiveTintColor: "#6B24AA", // Replace with your active color
        tabBarInactiveTintColor: "#84898c", // Replace with your inactive color
        tabBarStyle: {
          backgroundColor: "#DAD5F2", // Replace with your background color
        },
        tabBarLabelStyle: {
          fontSize: 12, // Replace with your size
          fontFamily: "SGB", // Replace with your font family
          marginBottom: 5,
        },
        headerShown: false,
      }}
      initialRouteName="Home"
    >
      <Tab.Screen
        name="기록"
        component={Tab1StackNavigator}
        options={{
          tabBarIcon: ({ color }) => (
            <Ionicons
              name="document-sharp"
              size={18}
              color={color}
              style={{ marginTop: 0 }}
            />
          ),
        }}
      />
      <Tab.Screen
        name="스트리밍"
        component={Tab2Screen}
        options={() => ({
          tabBarIcon: ({ color }) => (
            <Ionicons
              name="videocam-sharp"
              size={18}
              color={color}
              style={{ marginTop: 0 }}
            />
          ),
        })}
      />
      <Tab.Screen
        name="설정"
        component={Tab3StackNavigator}
        options={{
          tabBarIcon: ({ color }) => (
            <Ionicons
              name="settings-sharp"
              size={18}
              color={color}
              style={{ marginTop: 0 }}
            />
          ),
        }}
      />
    </Tab.Navigator>
  );
}
