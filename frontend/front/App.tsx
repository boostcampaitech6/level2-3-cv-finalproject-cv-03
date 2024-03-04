import React from 'react';
import { Text, View } from 'react-native';
import { useFonts } from 'expo-font';
import { NavigationContainer } from '@react-navigation/native';
import RootStackNavigator from './src/navigation/RootStackNavigator';
import { StyleSheet } from 'react-native'; // StyleSheet를 임포트합니다.
import { ThemeProvider, Button, createTheme } from '@rneui/themed';
import { UserProvider } from './UserContext';

const theme = createTheme({
  lightColors: {
    primary: '#FFA000'
  },
  mode: 'light',
  components: {
    Button: {
      radius: "10",
      style: {
        width: 200,
        marginVertical: 5,
      },
    },
    Text: {
      style: {
        fontFamily: 'SG',
      },
    },
    }
  },
);


export default function App() {
  let [fontsLoaded] = useFonts({
    'SG': require('./assets/fonts/SEBANG Gothic.ttf'), 
  });

  if (!fontsLoaded) {
    return <Text>Font Not Available</Text>;
  }

  return(
    <UserProvider>
      <ThemeProvider theme={theme}>
        <NavigationContainer>
          <RootStackNavigator />
        </NavigationContainer>
      </ThemeProvider>
    </UserProvider>
  );
}