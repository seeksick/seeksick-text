def test_log():
    try:
        print("테스트 시작")
        with open("test_log.txt", "w", encoding="utf-8") as f:
            f.write("테스트 로그 시작\n")
            f.flush()
        
        # 간단한 연산
        result = 1 + 1
        print(f"1 + 1 = {result}")
        
        with open("test_log.txt", "a", encoding="utf-8") as f:
            f.write(f"1 + 1 = {result}\n")
            f.flush()
            
        print("테스트 완료")
        
    except Exception as e:
        print(f"에러 발생: {str(e)}")
        with open("test_log.txt", "a", encoding="utf-8") as f:
            f.write(f"에러 발생: {str(e)}\n")
            f.flush()

if __name__ == "__main__":
    test_log() 