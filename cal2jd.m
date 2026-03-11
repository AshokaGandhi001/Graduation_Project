function jd = cal2jd(dateTimeMatrix)
    % 初始化儒略日向量
    jd = zeros(size(dateTimeMatrix, 1), 1);
    
    % 遍历每一行（每个日期时间）
    for i = 1:size(dateTimeMatrix, 1)
        year = dateTimeMatrix(i, 1);
        month = dateTimeMatrix(i, 2);
        day = dateTimeMatrix(i, 3);
        hour = dateTimeMatrix(i, 4);
        minute = dateTimeMatrix(i, 5);
        second = dateTimeMatrix(i, 6);
        
        % 检查月份是否小于3（即1月或2月），按儒略日转换规则调整
        if month <= 2
            year = year - 1;
            month = month + 12;
        end
        
        % 计算辅助值A和B
        A = floor(year / 100);
        B = 2 - A + floor(A / 4);
        
        % 计算儒略日的整数部分
        jd = floor(365.25 * (year + 4716)) + floor(30.6001 * (month + 1)) + day + B - 1524.5;
        
        % 将时间转换为日的小数部分
        timeFraction = (hour + minute / 60 + second / 3600) / 24;
        
        % 将时间的小数部分加到儒略日上
        jd(i) = jd + timeFraction;
    end
end