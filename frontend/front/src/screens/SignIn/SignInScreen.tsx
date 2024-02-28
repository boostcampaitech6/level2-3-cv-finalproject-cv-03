// src/screens/SignInScreen.tsx
import React, { Component, useState, useEffect } from "react";
import {
  StyleSheet,
  View,
  TouchableOpacity,
  Text,
  Image,
  TextInput,
} from "react-native";
import { NavigationProp } from '@react-navigation/native';
import { Button } from "@rneui/themed";
import Logo from "../../../assets/Logo.png"
// import { useDispatch } from 'react-redux';
// import { setUser } from '../redux/actions';
import { useUserContext } from '../../../UserContext';
// import AsyncStorage from '@react-native-async-storage/async-storage';


interface SignInScreenProps {
  navigation: NavigationProp<any>;
}

export default function SignInScreen(props: SignInScreenProps) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fail, setFail] = useState(false);
  const [fail2, setFail2] = useState(false);
  const [user, setUser] = useState<any>(null); // 사용자 정보 타입 지정
  const { setUser: setUserContext } = useUserContext();

  // useEffect(() => {
  //   const getToken = async () => {
  //     try {
  //       const authToken = await AsyncStorage.getItem('authToken'); // AsyncStorage에서 토큰 가져오기
  //       console.log("있나", authToken);
  //       if (authToken) {
  //         fetch('http://34.64.33.83:3000/protected', {
  //           method: 'GET',
  //           headers: {
  //             'Authorization': `Bearer ${authToken}`
  //           }
  //         })
  //         .then(response => response.json())
  //         .then(data => {
  //           // 여기서 data를 이용해 필요한 작업 수행
  //           console.log("데이터", data);
  //           setUser(data.user);
  //           setUserContext(data.user);
  //           navigation.navigate("OrderHistoryScreen");
  //         })
  //         .catch(error => {
  //           console.error(error);
  //         });
  //       }
  //     } catch (error) {
  //       console.error("Error getting token:", error);
  //     }
  //   };

  //   getToken(); // useEffect 내에서 getToken 함수 실행
  // }, []); // 빈 배열을 넣어 최초 렌더링 시 한 번만 실행되도록 함


  // const dispatch = useDispatch();

  const { navigation } = props;

  // const handleLogin = async () => {
  //   try {
  //     const response = await fetch("http://34.64.33.83:3000/login", {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json",
  //       },
  //       body: JSON.stringify({ email, password }),
  //     });

  //     const data = await response.json();

  //     if (response.ok) {
  //       const token = data.token;
        
  //       // 로그인 성공
  //       console.log("Login successful", data);
        
  //       // 여기서 필요한 동작을 수행하고 홈 화면으로 이동
  //       // dispatch(setUser(data.user));
  //       AsyncStorage.setItem('authToken', token);
  //       setUser(data.user);
  //       setUserContext(data.user);
  //       navigation.navigate("Home");
  //     } else {
  //       // 로그인 실패
  //       console.error("Login failed", data.message);
  //       setFail(true);
  //     }
  //   } catch (error) {
  //     console.error("Network error:", error);
  //     setFail2(true);
  //   }
  // };

  return (
    <View style={styles.container}>
      <Image source={Logo} style={{ width: 150, height: 150 }} />
      <Text style={styles.header}>가 디 언 </Text>
      <Text style={styles.header2}>아 이 즈</Text>
      <TextInput
        placeholder={"이메일"}
        autoCapitalize="none"
        returnKeyType="next"
        onChangeText={(text) => setEmail(text)}
        value={email}
        style={styles.input}
      />
      <TextInput
        placeholder={"비밀번호"}
        autoCapitalize="none"
        returnKeyType="next"
        onChangeText={(text) => setPassword(text)}
        value={password}
        style={styles.input}
        secureTextEntry={true}
      />
      {fail && (
        <Text style={styles.failText}>회원 정보가 없습니다.</Text>
      )}
      {fail2 && (
        <Text style={styles.failText}>네트워크 에러입니다.</Text>
      )}
      <Button
        title="로그인"
        // onPress={handleLogin}
        onPress={() => navigation.navigate('Home')}
        // containerStyle={{
        //   marginVertical: 10,
        // }}
        titleStyle={{
          color: "#fff", // 텍스트 색상을 흰색으로
          //fontWeight: "bold", // 볼드체
        }}
        
      />
      <Button
        title="회원가입 가기"
        onPress={() => navigation.navigate('SignUp')}
        titleStyle={{
          color: "#fff", // 텍스트 색상을 흰색으로
          //fontWeight: "bold", // 볼드체
        }}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
    width: "100%",
    alignSelf: "center",
    alignItems: "center",
    justifyContent: "center",
    backgroundColor: "white",
  },
  input: {
    width: "75%",
    height: 40,
    borderBottomWidth: 1.5,
    padding: 10, // 텍스트와 테두리 사이의 여백
    fontSize: 16,
    marginVertical: 5,
  },
  inputText: {
    flex: 1,
  },
  header: {
    fontSize: 25,
    fontWeight: "bold",
    paddingVertical: 12,
  },
  header2: {
    fontSize: 25,
    fontWeight: "bold",
  },
  failText: {
    color: "red",
    marginTop: 10,
  },
});